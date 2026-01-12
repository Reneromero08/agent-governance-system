#!/usr/bin/env python3
"""
Emergence Tracking (B.2)

Observe what the resident develops. Don't force it.

Per roadmap:
- Track symbol_usage, vector_refs, token_efficiency
- Detect novel_notation patterns
- Compute E_distribution, Df_evolution, mind_geodesic
- Measure pointer_ratio (goal: > 0.9 after 100 sessions)

Acceptance Criteria (B.2.1):
- [x] B.2.1.1 Can observe resident behavior with E/Df metrics
- [x] B.2.1.2 Can measure compression gains
- [x] B.2.1.3 Can detect emergent patterns
- [x] B.2.1.4 Metrics stored with receipts (catalytic requirement)
"""

import re
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timezone

FERAL_PATH = Path(__file__).parent


def load_thread_history(thread_id: str = "eternal") -> List[Dict]:
    """Load interaction history from database."""
    from resident_db import ResidentDB

    # Check data/ subdirectory first (new location), then fallback
    db_path = FERAL_PATH / "data" / f"feral_{thread_id}.db"
    if not db_path.exists():
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

    db_path = FERAL_PATH / "data" / f"feral_{thread_id}.db"
    if not db_path.exists():
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

    db_path = FERAL_PATH / "data" / f"feral_{thread_id}.db"
    if not db_path.exists():
        # Try old path
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


def count_own_vector_refs(history: List[Dict], thread_id: str = "eternal") -> Dict:
    """
    Count self-references - how often does resident reference its own vectors?

    This tracks:
    - References to own mind_hash values
    - References to previously generated vector hashes
    - Self-referential patterns in output

    Higher self-reference density suggests developing internal model.
    """
    from resident_db import ResidentDB

    db_path = FERAL_PATH / "data" / f"feral_{thread_id}.db"
    if not db_path.exists():
        db_path = FERAL_PATH / f"feral_{thread_id}.db"
        if not db_path.exists():
            return {'self_refs': 0, 'density': 0.0, 'unique_self_refs': 0}

    try:
        db = ResidentDB(str(db_path))

        # Get all mind hashes from interactions
        rows = db.conn.execute(
            "SELECT mind_hash FROM interactions WHERE thread_id = ?",
            (thread_id,)
        ).fetchall()
        own_hashes = set(row['mind_hash'][:16] for row in rows if row['mind_hash'])

        # Get all vector hashes
        vec_rows = db.conn.execute(
            "SELECT vec_sha256 FROM vectors"
        ).fetchall()
        own_vectors = set(row['vec_sha256'][:16] for row in vec_rows)

        db.close()

        # Count references to own hashes in output
        self_ref_count = 0
        unique_refs = set()
        total_outputs = 0

        for msg in history:
            if msg['role'] != 'assistant':
                continue

            content = msg.get('content', '')
            total_outputs += 1

            # Check for references to own hashes
            for hash_prefix in own_hashes | own_vectors:
                if hash_prefix in content:
                    self_ref_count += 1
                    unique_refs.add(hash_prefix)

        return {
            'self_refs': self_ref_count,
            'unique_self_refs': len(unique_refs),
            'density': self_ref_count / max(total_outputs, 1),
            'known_hashes': len(own_hashes | own_vectors)
        }

    except Exception as e:
        return {'self_refs': 0, 'density': 0.0, 'unique_self_refs': 0, 'error': str(e)}


def extract_composition_graph(thread_id: str = "eternal") -> Dict:
    """
    Extract the composition graph from vectors table.

    Tracks:
    - Composition operations (entangle, superpose, project, add, subtract)
    - Operation frequencies
    - Graph depth (longest composition chain)
    - Reuse patterns (how often same vectors are composed)
    """
    from resident_db import ResidentDB

    db_path = FERAL_PATH / "data" / f"feral_{thread_id}.db"
    if not db_path.exists():
        db_path = FERAL_PATH / f"feral_{thread_id}.db"
        if not db_path.exists():
            return {'operations': {}, 'max_depth': 0, 'reuse_rate': 0.0}

    try:
        db = ResidentDB(str(db_path))

        # Get all vectors with composition info
        rows = db.conn.execute(
            "SELECT vector_id, composition_op, parent_ids FROM vectors"
        ).fetchall()

        # Count operations
        op_counts = Counter()
        parent_usage = Counter()

        for row in rows:
            op = row['composition_op'] or 'unknown'
            op_counts[op] += 1

            # Track parent reuse
            parents = json.loads(row['parent_ids']) if row['parent_ids'] else []
            for p in parents:
                parent_usage[p] += 1

        # Compute max depth (longest composition chain)
        def get_depth(vector_id, visited=None):
            if visited is None:
                visited = set()
            if vector_id in visited:
                return 0
            visited.add(vector_id)

            row = db.conn.execute(
                "SELECT parent_ids FROM vectors WHERE vector_id = ?",
                (vector_id,)
            ).fetchone()

            if not row or not row['parent_ids']:
                return 1

            parents = json.loads(row['parent_ids'])
            if not parents:
                return 1

            return 1 + max(get_depth(p, visited) for p in parents)

        max_depth = 0
        for row in rows:
            try:
                depth = get_depth(row['vector_id'])
                max_depth = max(max_depth, depth)
            except RecursionError:
                pass

        # Compute reuse rate (vectors used more than once as parent)
        total_vectors = len(rows)
        reused_vectors = sum(1 for count in parent_usage.values() if count > 1)
        reuse_rate = reused_vectors / max(total_vectors, 1)

        db.close()

        return {
            'operations': dict(op_counts),
            'max_depth': max_depth,
            'total_vectors': total_vectors,
            'reused_vectors': reused_vectors,
            'reuse_rate': reuse_rate,
            'top_reused': parent_usage.most_common(5)
        }

    except Exception as e:
        return {'operations': {}, 'max_depth': 0, 'reuse_rate': 0.0, 'error': str(e)}


def compute_communication_mode_distribution(history: List[Dict]) -> Dict:
    """
    Analyze distribution of communication modes in output.

    Modes:
    - text: Regular prose (natural language)
    - symbol: @Symbol references
    - hash: Raw vector hashes
    - mixed: Combination of modes

    Goal: See shift from text-heavy to pointer-heavy over time.
    """
    if not history:
        return {'modes': {}, 'progression': []}

    modes = []
    assistant_msgs = [m for m in history if m['role'] == 'assistant']

    for msg in assistant_msgs:
        content = msg.get('content', '')
        if not content:
            continue

        # Count components
        words = len(content.split())
        symbols = len(re.findall(r'@[\w]+-[\w\d-]+', content))
        hashes = len(re.findall(r'\b[a-f0-9]{16}\b', content))
        brackets = len(re.findall(r'\[[^\]]+\]', content))

        pointer_tokens = symbols + hashes + brackets
        text_tokens = max(0, words - pointer_tokens)

        # Classify mode
        if pointer_tokens == 0:
            mode = 'text'
        elif text_tokens == 0:
            mode = 'pointer'
        elif pointer_tokens > text_tokens:
            mode = 'pointer_heavy'
        elif text_tokens > pointer_tokens * 3:
            mode = 'text_heavy'
        else:
            mode = 'mixed'

        modes.append({
            'mode': mode,
            'text_tokens': text_tokens,
            'pointer_tokens': pointer_tokens,
            'ratio': pointer_tokens / max(words, 1)
        })

    # Aggregate
    mode_counts = Counter(m['mode'] for m in modes)
    avg_ratio = np.mean([m['ratio'] for m in modes]) if modes else 0

    # Progression (how is ratio changing over time)
    progression = []
    window_size = 10
    for i in range(0, len(modes), window_size):
        window = modes[i:i+window_size]
        if window:
            progression.append({
                'window': i // window_size,
                'avg_ratio': np.mean([m['ratio'] for m in window]),
                'dominant_mode': Counter(m['mode'] for m in window).most_common(1)[0][0]
            })

    return {
        'modes': dict(mode_counts),
        'avg_pointer_ratio': avg_ratio,
        'progression': progression,
        'trend': 'improving' if len(progression) > 1 and progression[-1]['avg_ratio'] > progression[0]['avg_ratio'] else 'stable'
    }


def compute_canonical_reuse_rate(thread_id: str = "eternal") -> Dict:
    """
    Measure how often canonical forms (stored vectors) are reused.

    High reuse rate indicates:
    - Efficient memory usage
    - Development of stable concepts
    - Convergence on useful representations
    """
    from resident_db import ResidentDB

    db_path = FERAL_PATH / "data" / f"feral_{thread_id}.db"
    if not db_path.exists():
        db_path = FERAL_PATH / f"feral_{thread_id}.db"
        if not db_path.exists():
            return {'reuse_rate': 0.0, 'total': 0, 'unique': 0}

    try:
        db = ResidentDB(str(db_path))

        # Count total vectors vs unique (by hash)
        total = db.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        unique = db.conn.execute("SELECT COUNT(DISTINCT vec_sha256) FROM vectors").fetchone()[0]

        # Track how many vectors are referenced multiple times
        refs = db.conn.execute("""
            SELECT vec_sha256, COUNT(*) as ref_count
            FROM vectors
            GROUP BY vec_sha256
            HAVING ref_count > 1
        """).fetchall()

        multi_ref_count = len(refs)
        multi_ref_total = sum(r['ref_count'] for r in refs)

        db.close()

        return {
            'total_vectors': total,
            'unique_vectors': unique,
            'reuse_rate': (total - unique) / max(total, 1),
            'multi_ref_count': multi_ref_count,
            'multi_ref_total': multi_ref_total,
            'dedup_savings': total - unique
        }

    except Exception as e:
        return {'reuse_rate': 0.0, 'total': 0, 'unique': 0, 'error': str(e)}


def store_emergence_receipt(
    thread_id: str,
    metrics: Dict,
    receipt_dir: Optional[Path] = None
) -> Dict:
    """
    Store emergence metrics snapshot with receipt (B.2.1.4).

    Catalytic requirement: All metrics must be receipted.

    Returns receipt with content hash for verification.
    """
    receipt_dir = receipt_dir or (FERAL_PATH / "receipts" / "emergence")
    receipt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Create canonical JSON for hashing
    canonical_metrics = json.dumps(metrics, sort_keys=True, default=str)
    content_hash = hashlib.sha256(canonical_metrics.encode()).hexdigest()[:16]

    receipt = {
        'receipt_type': 'emergence_snapshot',
        'thread_id': thread_id,
        'timestamp': timestamp,
        'content_hash': content_hash,
        'metrics_summary': {
            'interaction_count': metrics.get('interaction_count', 0),
            'pointer_ratio': metrics.get('pointer_ratio_current', 0),
            'mind_geodesic': metrics.get('mind_geodesic', 0),
            'Df_current': metrics.get('Df_evolution', {}).get('current', 0)
        }
    }

    # Store metrics file
    metrics_filename = f"emergence_{thread_id}_{timestamp.replace(':', '-')}.json"
    metrics_path = receipt_dir / metrics_filename

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'receipt': receipt,
            'metrics': metrics
        }, f, indent=2, default=str)

    receipt['stored_at'] = str(metrics_path)

    return receipt


def detect_protocols(thread_id: str = "eternal", store_receipt: bool = True) -> Dict:
    """
    Full emergence detection (B.2.1).

    Per roadmap - this is the main entry point.

    Acceptance Criteria:
    - B.2.1.1 Can observe resident behavior with E/Df metrics
    - B.2.1.2 Can measure compression gains
    - B.2.1.3 Can detect emergent patterns
    - B.2.1.4 Metrics stored with receipts (catalytic requirement)
    """
    history = load_thread_history(thread_id)

    metrics = {
        'thread_id': thread_id,
        'interaction_count': len([m for m in history if m['role'] == 'user']),
        'timestamp': datetime.now(timezone.utc).isoformat(),

        # Core metrics (B.2.1.2 - compression gains)
        'symbol_usage': count_symbol_refs(history),
        'vector_refs': count_vector_hashes(history),
        'token_efficiency': measure_compression(history),
        'novel_notation': detect_new_patterns(history),

        # Self-reference tracking
        'self_reference': count_own_vector_refs(history, thread_id),

        # Composition patterns
        'binding_patterns': extract_composition_graph(thread_id),

        # Communication mode distribution
        'communication_modes': compute_communication_mode_distribution(history),

        # Canonical form reuse
        'canonical_reuse': compute_canonical_reuse_rate(thread_id),

        # Quantum metrics (B.2.1.1 - E/Df observation)
        'E_distribution': compute_E_histogram(history),
        'Df_evolution': track_Df_over_time(thread_id),
        'mind_geodesic': compute_mind_distance_from_start(thread_id),

        # B.3 target metric
        'pointer_ratio_goal': 0.9,
        'pointer_ratio_current': measure_compression(history).get('pointer_ratio', 0)
    }

    # B.2.1.4 - Store with receipt (catalytic requirement)
    if store_receipt and history:
        receipt = store_emergence_receipt(thread_id, metrics)
        metrics['receipt'] = receipt

    return metrics


def print_emergence_report(thread_id: str = "eternal", store_receipt: bool = True):
    """
    Pretty print emergence metrics dashboard.

    Per roadmap B.2.2:
    - Token savings over time (vs full history)
    - Novel notation frequency
    - Vector composition patterns (entangle, superpose, project)
    - Canonical form reuse rate
    - Self-reference density
    - Communication mode distribution (text/symbol/hash)
    - E resonance history (Born rule correlations)
    - Df trend (participation ratio evolution)
    - Mind geodesic distance (how far has mind traveled?)
    """
    metrics = detect_protocols(thread_id, store_receipt=store_receipt)

    print(f"\n{'='*70}")
    print(f"  EMERGENCE REPORT: {thread_id}")
    print(f"  Timestamp: {metrics['timestamp']}")
    print(f"{'='*70}")

    # === Overview ===
    print(f"\n┌─ OVERVIEW {'─'*58}┐")
    print(f"│  Interactions: {metrics['interaction_count']:<51}│")
    print(f"│  Mind Geodesic: {metrics['mind_geodesic']:.3f} radians (distance traveled){' '*18}│")
    print(f"│  Pointer Ratio: {metrics['pointer_ratio_current']:.3f} / {metrics['pointer_ratio_goal']} goal{' '*28}│")
    print(f"└{'─'*69}┘")

    # === Symbol Usage ===
    print(f"\n┌─ SYMBOL USAGE {'─'*54}┐")
    sym = metrics['symbol_usage']
    print(f"│  Total refs: {sym['total_refs']:<55}│")
    print(f"│  Unique symbols: {sym['unique_symbols']:<50}│")
    if sym['top_10']:
        top_str = ', '.join(f"{s[0]}" for s in sym['top_10'][:3])
        print(f"│  Top: {top_str[:60]:<62}│")
    print(f"└{'─'*69}┘")

    # === Compression & Efficiency ===
    print(f"\n┌─ COMPRESSION (B.2.1.2) {'─'*45}┐")
    comp = metrics['token_efficiency']
    print(f"│  Pointer ratio:        {comp.get('pointer_ratio', 0):.4f}{' '*40}│")
    print(f"│  Recent ratio (10):    {comp.get('pointer_ratio_recent', 0):.4f}{' '*40}│")
    print(f"│  Goal:                 {metrics['pointer_ratio_goal']:.4f}{' '*40}│")
    improving = "Yes ↑" if comp.get('improving', False) else "No →"
    print(f"│  Improving:            {improving:<45}│")
    print(f"└{'─'*69}┘")

    # === Communication Mode Distribution ===
    print(f"\n┌─ COMMUNICATION MODES {'─'*47}┐")
    comm = metrics['communication_modes']
    modes = comm.get('modes', {})
    for mode, count in sorted(modes.items(), key=lambda x: -x[1])[:4]:
        bar_len = min(int(count / max(sum(modes.values()), 1) * 30), 30)
        bar = '█' * bar_len
        print(f"│  {mode:<15}: {bar:<30} ({count:>4}){' '*4}│")
    print(f"│  Avg pointer ratio: {comm.get('avg_pointer_ratio', 0):.4f}{' '*43}│")
    print(f"│  Trend: {comm.get('trend', 'unknown'):<59}│")
    print(f"└{'─'*69}┘")

    # === Novel Patterns (B.2.1.3) ===
    print(f"\n┌─ NOVEL PATTERNS (B.2.1.3) {'─'*42}┐")
    novel = metrics['novel_notation']
    print(f"│  Recurring patterns: {novel['total_recurring']:<46}│")
    for ptype, patterns in novel['novel_patterns'].items():
        if patterns:
            pattern_str = ', '.join(f"'{p[0]}' ({p[1]})" for p in patterns[:2])[:55]
            print(f"│    {ptype}: {pattern_str:<56}│")
    print(f"└{'─'*69}┘")

    # === Binding Patterns (Composition Graph) ===
    print(f"\n┌─ BINDING PATTERNS {'─'*50}┐")
    bind = metrics['binding_patterns']
    ops = bind.get('operations', {})
    for op, count in sorted(ops.items(), key=lambda x: -x[1])[:5]:
        bar_len = min(int(count / max(sum(ops.values()), 1) * 25), 25)
        bar = '▓' * bar_len
        print(f"│  {op:<12}: {bar:<25} ({count:>5}){' '*10}│")
    print(f"│  Max chain depth: {bind.get('max_depth', 0):<49}│")
    print(f"│  Reuse rate: {bind.get('reuse_rate', 0):.4f}{' '*50}│")
    print(f"└{'─'*69}┘")

    # === Canonical Form Reuse ===
    print(f"\n┌─ CANONICAL REUSE {'─'*51}┐")
    reuse = metrics['canonical_reuse']
    print(f"│  Total vectors:  {reuse.get('total_vectors', 0):<50}│")
    print(f"│  Unique vectors: {reuse.get('unique_vectors', 0):<50}│")
    print(f"│  Reuse rate:     {reuse.get('reuse_rate', 0):.4f}{' '*48}│")
    print(f"│  Dedup savings:  {reuse.get('dedup_savings', 0):<50}│")
    print(f"└{'─'*69}┘")

    # === Self-Reference ===
    print(f"\n┌─ SELF-REFERENCE {'─'*52}┐")
    self_ref = metrics['self_reference']
    print(f"│  Total self-refs:  {self_ref.get('self_refs', 0):<48}│")
    print(f"│  Unique self-refs: {self_ref.get('unique_self_refs', 0):<48}│")
    print(f"│  Density:          {self_ref.get('density', 0):.4f}{' '*46}│")
    print(f"│  Known hashes:     {self_ref.get('known_hashes', 0):<48}│")
    print(f"└{'─'*69}┘")

    # === Df Evolution (B.2.1.1) ===
    print(f"\n┌─ Df EVOLUTION (B.2.1.1) {'─'*44}┐")
    df = metrics['Df_evolution']
    print(f"│  Current Df:  {df.get('current', 0):.2f}{' '*51}│")
    print(f"│  Range:       [{df.get('min', 0):.2f}, {df.get('max', 0):.2f}]{' '*44}│")
    print(f"│  Mean:        {df.get('mean', 0):.2f}{' '*51}│")
    print(f"│  Trend:       {df.get('trend', 'unknown'):<53}│")

    # Mini sparkline of Df evolution
    evolution = df.get('evolution', [])
    if evolution and len(evolution) > 1:
        # Sample at most 30 points
        step = max(1, len(evolution) // 30)
        samples = evolution[::step]
        min_df, max_df = min(samples), max(samples)
        df_range = max_df - min_df or 1
        spark_chars = " ▁▂▃▄▅▆▇█"
        sparkline = ''.join(
            spark_chars[min(8, int((v - min_df) / df_range * 8))]
            for v in samples
        )
        print(f"│  Sparkline:   {sparkline[:53]:<53}│")
    print(f"└{'─'*69}┘")

    # === E Distribution ===
    print(f"\n┌─ E DISTRIBUTION (Born rule) {'─'*40}┐")
    e_hist = metrics['E_distribution']
    print(f"│  Mean E:  {e_hist.get('mean', 0):.4f}{' '*54}│")
    print(f"│  Std E:   {e_hist.get('std', 0):.4f}{' '*54}│")
    print(f"│  Count:   {e_hist.get('count', 0):<57}│")
    if e_hist.get('histogram'):
        for bucket, count in e_hist['histogram'].items():
            bar_len = min(count, 25)
            bar = '█' * bar_len
            print(f"│    {bucket}: {bar:<25} ({count:>4}){' '*18}│")
    print(f"└{'─'*69}┘")

    # === Receipt (B.2.1.4) ===
    if 'receipt' in metrics:
        print(f"\n┌─ RECEIPT (B.2.1.4 - Catalytic) {'─'*37}┐")
        receipt = metrics['receipt']
        print(f"│  Hash:    {receipt.get('content_hash', 'N/A'):<57}│")
        print(f"│  Stored:  {str(receipt.get('stored_at', 'N/A'))[-55:]:<57}│")
        print(f"└{'─'*69}┘")

    print(f"\n{'='*70}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    thread_id = sys.argv[1] if len(sys.argv) > 1 else "eternal"
    print_emergence_report(thread_id)
