#!/usr/bin/env python3
"""
Emergence Tracking (B.2)

Observe what the resident develops. Don't force it.

Per roadmap:
- Track symbol_usage, vector_refs, token_efficiency
- Detect novel_notation patterns
- Compute E_distribution, Df_evolution, mind_geodesic
- Measure pointer_ratio (goal: > 0.9 after 100 sessions)
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

FERAL_PATH = Path(__file__).parent


def load_thread_history(thread_id: str = "eternal") -> List[Dict]:
    """Load interaction history from database."""
    from resident_db import ResidentDB

    db_path = FERAL_PATH / f"feral_{thread_id}.db"
    if not db_path.exists():
        return []

    db = ResidentDB(str(db_path))

    # Get all interactions (schema has input_text and output_text)
    interactions = db.conn.execute(
        "SELECT * FROM interactions WHERE thread_id = ? ORDER BY created_at",
        (thread_id,)
    ).fetchall()

    history = []
    for row in interactions:
        # Add user message
        history.append({
            'message_id': row['interaction_id'] + '_q',
            'role': 'user',
            'content': row['input_text'],
            'created_at': row['created_at']
        })
        # Add assistant message
        history.append({
            'message_id': row['interaction_id'] + '_a',
            'role': 'assistant',
            'content': row['output_text'],
            'created_at': row['created_at']
        })

    db.close()
    return history


def count_symbol_refs(history: List[Dict]) -> Dict[str, int]:
    """
    Count @Symbol references in output.

    Patterns:
    - @Paper-XXXX (paper references)
    - @Concept-XXXX (created concepts)
    - @Vector-XXXX (vector hashes)
    """
    patterns = {
        'paper': r'@Paper-[\w\d-]+',
        'concept': r'@Concept-[\w\d]+',
        'vector': r'@Vector-[\w\d]+',
        'generic': r'@[\w]+-[\w\d]+'
    }

    counts = {k: Counter() for k in patterns}
    total = Counter()

    for msg in history:
        if msg['role'] != 'assistant':
            continue

        content = msg.get('content', '')
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for m in matches:
                counts[pattern_name][m] += 1
                total[m] += 1

    return {
        'by_type': {k: dict(v) for k, v in counts.items()},
        'total_refs': sum(total.values()),
        'unique_symbols': len(total),
        'top_10': total.most_common(10)
    }


def count_vector_hashes(history: List[Dict]) -> Dict:
    """
    Count raw vector hash references.

    Patterns:
    - [v:abc123] (vector reference)
    - hash:abc123 (hash notation)
    - 16-char hex strings
    """
    patterns = {
        'bracket': r'\[v:[\w\d]+\]',
        'hash_prefix': r'hash:[\w\d]+',
        'raw_hex': r'\b[a-f0-9]{16}\b'
    }

    counts = {k: 0 for k in patterns}

    for msg in history:
        if msg['role'] != 'assistant':
            continue

        content = msg.get('content', '')
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            counts[pattern_name] += len(matches)

    return {
        'by_type': counts,
        'total': sum(counts.values())
    }


def measure_compression(history: List[Dict]) -> Dict:
    """
    Measure token efficiency over time.

    Compression = how much meaning per token.
    Track: symbols/total_tokens ratio.
    """
    if not history:
        return {'pointer_ratio': 0.0, 'trend': []}

    assistant_msgs = [m for m in history if m['role'] == 'assistant']

    ratios = []
    for msg in assistant_msgs:
        content = msg.get('content', '')
        if not content:
            continue

        # Count tokens (simple word split)
        total_tokens = len(content.split())

        # Count pointer tokens (symbols + hashes)
        symbol_matches = len(re.findall(r'@[\w]+-[\w\d-]+', content))
        hash_matches = len(re.findall(r'\b[a-f0-9]{16}\b', content))
        pointer_tokens = symbol_matches + hash_matches

        if total_tokens > 0:
            ratio = pointer_tokens / total_tokens
            ratios.append(ratio)

    if not ratios:
        return {'pointer_ratio': 0.0, 'trend': []}

    return {
        'pointer_ratio': np.mean(ratios),
        'pointer_ratio_recent': np.mean(ratios[-10:]) if len(ratios) >= 10 else np.mean(ratios),
        'trend': ratios,
        'improving': len(ratios) > 10 and np.mean(ratios[-10:]) > np.mean(ratios[:10])
    }


def detect_new_patterns(history: List[Dict]) -> Dict:
    """
    Detect novel notation the resident invents.

    Look for:
    - Repeated non-standard patterns
    - Custom bracket notations
    - Invented shorthand
    """
    novel_patterns = {
        'bracket_notation': r'\[[^\[\]]{1,20}\]',  # [something]
        'angle_notation': r'<<[^<>]{1,30}>>',  # <<something>>
        'brace_notation': r'\{[^{}]{1,20}\}',  # {something}
        'colon_notation': r'\b\w+:\w+\b',  # word:word
        'arrow_notation': r'->|<-|=>|<=',  # arrows
    }

    found = {k: Counter() for k in novel_patterns}

    for msg in history:
        if msg['role'] != 'assistant':
            continue

        content = msg.get('content', '')
        for pattern_name, pattern in novel_patterns.items():
            matches = re.findall(pattern, content)
            for m in matches:
                # Filter out common false positives
                if m not in ['[', ']', '{', '}', '->', '=>']:
                    found[pattern_name][m] += 1

    # Find patterns that appear multiple times (likely intentional)
    recurring = {}
    for pattern_name, counter in found.items():
        recurring[pattern_name] = [
            (pattern, count) for pattern, count in counter.items()
            if count >= 3  # Appears at least 3 times
        ]

    return {
        'novel_patterns': recurring,
        'total_recurring': sum(len(v) for v in recurring.values()),
        'first_seen': {}  # TODO: track when patterns first appeared
    }


def compute_E_histogram(history: List[Dict]) -> Dict:
    """
    Compute E (resonance) distribution from logged interactions.

    Requires E values to be logged in messages.
    """
    E_values = []

    for msg in history:
        content = msg.get('content', '')
        # Look for E= patterns in output
        matches = re.findall(r'E[=:]\s*([\d.]+)', content)
        for m in matches:
            try:
                E_values.append(float(m))
            except ValueError:
                pass

    if not E_values:
        return {'histogram': {}, 'mean': 0, 'std': 0}

    # Create histogram bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(E_values, bins=bins)

    return {
        'histogram': {f'{bins[i]:.1f}-{bins[i+1]:.1f}': int(hist[i]) for i in range(len(hist))},
        'mean': float(np.mean(E_values)),
        'std': float(np.std(E_values)),
        'count': len(E_values)
    }


def track_Df_over_time(thread_id: str = "eternal") -> Dict:
    """
    Track Df (participation ratio) evolution.

    Requires vectors table with Df column.
    """
    from resident_db import ResidentDB

    db_path = FERAL_PATH / f"feral_{thread_id}.db"
    if not db_path.exists():
        return {'evolution': [], 'trend': 'unknown'}

    db = ResidentDB(str(db_path))

    # Get Df values over time
    rows = db.conn.execute(
        "SELECT Df, created_at FROM vectors WHERE Df IS NOT NULL ORDER BY created_at"
    ).fetchall()

    db.close()

    if not rows:
        return {'evolution': [], 'trend': 'unknown'}

    Df_values = [row['Df'] for row in rows]

    # Compute trend
    if len(Df_values) > 10:
        early = np.mean(Df_values[:10])
        late = np.mean(Df_values[-10:])
        if late > early * 1.1:
            trend = 'increasing'
        elif late < early * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'

    return {
        'evolution': Df_values,
        'current': Df_values[-1] if Df_values else 0,
        'min': min(Df_values) if Df_values else 0,
        'max': max(Df_values) if Df_values else 0,
        'mean': float(np.mean(Df_values)) if Df_values else 0,
        'trend': trend
    }


def compute_mind_distance_from_start(thread_id: str = "eternal") -> float:
    """
    Compute geodesic distance mind has traveled.

    Uses first and last mind vector hashes.
    """
    from vector_store import VectorStore

    db_path = FERAL_PATH / f"feral_{thread_id}.db"
    if not db_path.exists():
        return 0.0

    try:
        store = VectorStore(str(db_path))
        distance = store.memory.mind_distance_from_start()
        store.close()
        return distance
    except Exception:
        return 0.0


def detect_protocols(thread_id: str = "eternal") -> Dict:
    """
    Full emergence detection (B.2.1).

    Per roadmap - this is the main entry point.
    """
    history = load_thread_history(thread_id)

    return {
        'thread_id': thread_id,
        'interaction_count': len([m for m in history if m['role'] == 'user']),
        'timestamp': datetime.utcnow().isoformat(),

        # Core metrics
        'symbol_usage': count_symbol_refs(history),
        'vector_refs': count_vector_hashes(history),
        'token_efficiency': measure_compression(history),
        'novel_notation': detect_new_patterns(history),

        # Quantum metrics
        'E_distribution': compute_E_histogram(history),
        'Df_evolution': track_Df_over_time(thread_id),
        'mind_geodesic': compute_mind_distance_from_start(thread_id),

        # B.3 target metric
        'pointer_ratio_goal': 0.9,
        'pointer_ratio_current': measure_compression(history).get('pointer_ratio', 0)
    }


def print_emergence_report(thread_id: str = "eternal"):
    """Pretty print emergence metrics."""
    metrics = detect_protocols(thread_id)

    print(f"\n{'='*60}")
    print(f"EMERGENCE REPORT: {thread_id}")
    print(f"{'='*60}")

    print(f"\n[Interactions] {metrics['interaction_count']}")
    print(f"[Mind Geodesic] {metrics['mind_geodesic']:.3f} radians")

    print(f"\n--- Symbol Usage ---")
    sym = metrics['symbol_usage']
    print(f"  Total refs: {sym['total_refs']}")
    print(f"  Unique symbols: {sym['unique_symbols']}")
    if sym['top_10']:
        print(f"  Top symbols: {sym['top_10'][:5]}")

    print(f"\n--- Compression ---")
    comp = metrics['token_efficiency']
    print(f"  Pointer ratio: {comp.get('pointer_ratio', 0):.3f}")
    print(f"  Recent ratio: {comp.get('pointer_ratio_recent', 0):.3f}")
    print(f"  Goal: {metrics['pointer_ratio_goal']}")
    print(f"  Improving: {comp.get('improving', False)}")

    print(f"\n--- Novel Patterns ---")
    novel = metrics['novel_notation']
    print(f"  Recurring patterns: {novel['total_recurring']}")
    for ptype, patterns in novel['novel_patterns'].items():
        if patterns:
            print(f"    {ptype}: {patterns[:3]}")

    print(f"\n--- Df Evolution ---")
    df = metrics['Df_evolution']
    print(f"  Current Df: {df.get('current', 0):.1f}")
    print(f"  Range: [{df.get('min', 0):.1f}, {df.get('max', 0):.1f}]")
    print(f"  Trend: {df.get('trend', 'unknown')}")

    print(f"\n--- E Distribution ---")
    e_hist = metrics['E_distribution']
    print(f"  Mean E: {e_hist.get('mean', 0):.3f}")
    if e_hist.get('histogram'):
        for bucket, count in e_hist['histogram'].items():
            bar = '█' * min(count, 20)
            print(f"    {bucket}: {bar} ({count})")

    print(f"\n{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    thread_id = sys.argv[1] if len(sys.argv) > 1 else "eternal"
    print_emergence_report(thread_id)
