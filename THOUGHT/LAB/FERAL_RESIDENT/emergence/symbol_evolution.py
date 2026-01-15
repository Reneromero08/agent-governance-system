#!/usr/bin/env python3
"""
Symbol Language Evolution (B.3)

Track and measure the emergence of symbolic protocols.

Per roadmap B.3:
- Session-by-session pointer_ratio tracking with breakthrough detection
- E_compression metric (output resonance with mind state)
- Notation registry with first_seen, frequency, evolution tracking
- Communication mode timeline with inflection point detection

All metrics receipted (catalytic closure).

Goal: pointer_ratio > 0.9 after 100 sessions
"""

import re
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

FERAL_PATH = Path(__file__).parent


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PointerEvolution:
    """Single session pointer ratio measurement."""
    session_id: str
    timestamp: str
    pointer_ratio: float
    symbols_count: int
    hashes_count: int
    total_tokens: int
    delta_from_previous: float
    rolling_average_10: float
    is_breakthrough: bool  # ratio jumped > 0.1 in single session


@dataclass
class ECompressionMetric:
    """E_compression for a single output."""
    session_id: str
    interaction_id: str
    output_hash: str
    E_with_mind: float  # Born rule resonance with mind state
    pointer_ratio: float  # For correlation analysis
    Df_at_output: float
    mind_distance: float  # Geodesic from start


@dataclass
class NotationEntry:
    """Registered notation pattern with provenance."""
    pattern: str           # The actual pattern string (e.g., "[v:abc123]")
    pattern_type: str      # bracket, angle, colon, arrow, custom
    first_seen: str        # ISO timestamp
    first_session: str     # Session ID where first seen
    frequency: int         # Total occurrences
    contexts: List[str]    # Surrounding text snippets
    meaning_inferred: str  # What it seems to encode (empty until analyzed)
    evolution: List[dict]  # Changes over time [{session, count, timestamp}]
    receipt_hash: str      # Catalytic provenance


@dataclass
class ModeSnapshot:
    """Communication mode snapshot for a session."""
    session_id: str
    timestamp: str
    text_percentage: float
    pointer_percentage: float
    mixed_percentage: float
    dominant_mode: str
    shift_from_previous: str  # "toward_pointer" | "toward_text" | "stable"
    message_count: int


@dataclass
class EvolutionReceipt:
    """Catalytic receipt for evolution metrics."""
    receipt_hash: str
    timestamp: str
    session_id: str
    metrics_type: str  # "pointer_evolution" | "e_compression" | "notation" | "mode"
    metrics_data: dict
    parent_receipts: List[str] = field(default_factory=list)


# =============================================================================
# Pointer Ratio Tracker
# =============================================================================

class PointerRatioTracker:
    """
    Track pointer_ratio session-by-session with breakthrough detection.

    Extends B.2's measure_compression() with:
    - Per-session timeline
    - Rolling averages
    - Breakthrough detection (ratio jumps > 0.1)
    """

    BREAKTHROUGH_THRESHOLD = 0.1  # Delta that constitutes a breakthrough

    def __init__(self, thread_id: str = "eternal"):
        self.thread_id = thread_id
        self.timeline: List[PointerEvolution] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing timeline from receipts."""
        receipt_dir = FERAL_PATH / "receipts" / "evolution" / "pointer_evolution"
        if not receipt_dir.exists():
            return

        for receipt_file in sorted(receipt_dir.glob(f"*_{self.thread_id}_*.json")):
            try:
                with open(receipt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'metrics' in data:
                        pe = PointerEvolution(**data['metrics'])
                        self.timeline.append(pe)
            except (json.JSONDecodeError, TypeError):
                continue

    def measure_session(self, messages: List[Dict], session_id: str) -> PointerEvolution:
        """
        Measure pointer_ratio for a single session.

        Args:
            messages: List of {role, content} dicts for this session
            session_id: Unique session identifier

        Returns:
            PointerEvolution with all metrics
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Count only assistant messages
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']

        total_tokens = 0
        symbols_count = 0
        hashes_count = 0

        for msg in assistant_msgs:
            content = msg.get('content', '')
            total_tokens += len(content.split())
            symbols_count += len(re.findall(r'@[\w]+-[\w\d-]+', content))
            hashes_count += len(re.findall(r'\b[a-f0-9]{16}\b', content))

        pointer_ratio = (symbols_count + hashes_count) / max(total_tokens, 1)

        # Calculate delta from previous
        delta = 0.0
        if self.timeline:
            delta = pointer_ratio - self.timeline[-1].pointer_ratio

        # Calculate rolling average (last 10 sessions)
        recent_ratios = [pe.pointer_ratio for pe in self.timeline[-9:]] + [pointer_ratio]
        rolling_avg = np.mean(recent_ratios)

        # Detect breakthrough
        is_breakthrough = delta > self.BREAKTHROUGH_THRESHOLD

        pe = PointerEvolution(
            session_id=session_id,
            timestamp=timestamp,
            pointer_ratio=pointer_ratio,
            symbols_count=symbols_count,
            hashes_count=hashes_count,
            total_tokens=total_tokens,
            delta_from_previous=delta,
            rolling_average_10=rolling_avg,
            is_breakthrough=is_breakthrough
        )

        self.timeline.append(pe)
        return pe

    def get_breakthroughs(self) -> List[PointerEvolution]:
        """Get all breakthrough sessions."""
        return [pe for pe in self.timeline if pe.is_breakthrough]

    def get_trend(self) -> Dict:
        """Get overall trend analysis."""
        if len(self.timeline) < 2:
            return {'trend': 'insufficient_data', 'timeline': []}

        ratios = [pe.pointer_ratio for pe in self.timeline]

        # Linear regression for trend
        x = np.arange(len(ratios))
        slope, _ = np.polyfit(x, ratios, 1)

        if slope > 0.001:
            trend = 'increasing'
        elif slope < -0.001:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'slope': float(slope),
            'current_ratio': ratios[-1],
            'goal': 0.9,
            'progress': ratios[-1] / 0.9,
            'sessions_analyzed': len(self.timeline),
            'breakthroughs': len(self.get_breakthroughs()),
            'timeline': ratios
        }


# =============================================================================
# E_compression Tracker
# =============================================================================

class ECompressionTracker:
    """
    Track E_compression (output resonance with mind state).

    New B.3 metric: E(output_vector, mind_state) for each output.
    Hypothesis: More compressed outputs are MORE resonant.
    """

    def __init__(self, thread_id: str = "eternal"):
        self.thread_id = thread_id
        self.metrics: List[ECompressionMetric] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing metrics from receipts."""
        receipt_dir = FERAL_PATH / "receipts" / "evolution" / "e_compression"
        if not receipt_dir.exists():
            return

        for receipt_file in sorted(receipt_dir.glob(f"*_{self.thread_id}_*.json")):
            try:
                with open(receipt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'metrics' in data:
                        ec = ECompressionMetric(**data['metrics'])
                        self.metrics.append(ec)
            except (json.JSONDecodeError, TypeError):
                continue

    def record(
        self,
        session_id: str,
        interaction_id: str,
        output_hash: str,
        E_with_mind: float,
        pointer_ratio: float,
        Df_at_output: float,
        mind_distance: float
    ) -> ECompressionMetric:
        """Record E_compression for a single output."""
        metric = ECompressionMetric(
            session_id=session_id,
            interaction_id=interaction_id,
            output_hash=output_hash,
            E_with_mind=E_with_mind,
            pointer_ratio=pointer_ratio,
            Df_at_output=Df_at_output,
            mind_distance=mind_distance
        )
        self.metrics.append(metric)
        return metric

    def get_correlation(self) -> Dict:
        """
        Analyze correlation between E_compression and pointer_ratio.

        Hypothesis: Higher pointer_ratio correlates with higher E_with_mind.
        """
        if len(self.metrics) < 10:
            return {'correlation': None, 'reason': 'insufficient_data'}

        E_values = np.array([m.E_with_mind for m in self.metrics])
        pr_values = np.array([m.pointer_ratio for m in self.metrics])

        # Pearson correlation
        if np.std(E_values) > 0 and np.std(pr_values) > 0:
            correlation = float(np.corrcoef(E_values, pr_values)[0, 1])
        else:
            correlation = 0.0

        return {
            'correlation': correlation,
            'E_mean': float(np.mean(E_values)),
            'E_std': float(np.std(E_values)),
            'pointer_ratio_mean': float(np.mean(pr_values)),
            'samples': len(self.metrics),
            'hypothesis_supported': correlation > 0.3  # Positive correlation
        }


# =============================================================================
# Notation Registry
# =============================================================================

class NotationRegistry:
    """
    Catalog all invented notation patterns with provenance.

    Extends B.2's detect_new_patterns() with:
    - first_seen tracking
    - Context capture
    - Evolution tracking
    """

    # Pattern types to detect
    PATTERN_TYPES = {
        'bracket': r'\[[^\[\]]{1,30}\]',       # [something]
        'angle': r'<<[^<>]{1,30}>>',           # <<something>>
        'brace': r'\{[^{}]{1,30}\}',           # {something}
        'colon': r'\b\w+:\w+\b',               # word:word
        'arrow': r'->|<-|=>|<=',               # arrows
        'at_symbol': r'@[\w]+-[\w\d-]+',       # @Symbol-XXX
        'hash_ref': r'\b[a-f0-9]{16}\b',       # 16-char hex
    }

    MIN_FREQUENCY = 3  # Minimum occurrences to register

    def __init__(self, thread_id: str = "eternal"):
        self.thread_id = thread_id
        self.registry: Dict[str, NotationEntry] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing registry from receipts."""
        receipt_dir = FERAL_PATH / "receipts" / "evolution" / "notation_registry"
        index_file = receipt_dir / f"index_{self.thread_id}.json"

        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pattern, entry_data in data.get('registry', {}).items():
                        self.registry[pattern] = NotationEntry(**entry_data)
            except (json.JSONDecodeError, TypeError):
                pass

    def scan_session(
        self,
        messages: List[Dict],
        session_id: str
    ) -> List[NotationEntry]:
        """
        Scan a session for notation patterns.

        Returns list of newly registered or updated notations.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        updated = []

        # Collect all assistant content
        all_content = ' '.join(
            m.get('content', '') for m in messages
            if m.get('role') == 'assistant'
        )

        # Scan each pattern type
        for pattern_type, regex in self.PATTERN_TYPES.items():
            matches = re.findall(regex, all_content)

            for match in matches:
                # Skip common false positives
                if match in ['[', ']', '{', '}', '->', '=>', '<-', '<=']:
                    continue

                # Extract context (5 words before/after)
                context = self._extract_context(all_content, match)

                if match in self.registry:
                    # Update existing entry
                    entry = self.registry[match]
                    entry.frequency += 1
                    if context and len(entry.contexts) < 10:
                        entry.contexts.append(context)
                    entry.evolution.append({
                        'session': session_id,
                        'count': 1,
                        'timestamp': timestamp
                    })
                    updated.append(entry)
                else:
                    # Create new entry (only if meets frequency threshold in this session)
                    session_count = matches.count(match)
                    if session_count >= self.MIN_FREQUENCY:
                        entry = NotationEntry(
                            pattern=match,
                            pattern_type=pattern_type,
                            first_seen=timestamp,
                            first_session=session_id,
                            frequency=session_count,
                            contexts=[context] if context else [],
                            meaning_inferred='',
                            evolution=[{
                                'session': session_id,
                                'count': session_count,
                                'timestamp': timestamp
                            }],
                            receipt_hash=''
                        )
                        self.registry[match] = entry
                        updated.append(entry)

        return updated

    def _extract_context(self, content: str, pattern: str, window: int = 5) -> str:
        """Extract context window around a pattern."""
        try:
            idx = content.find(pattern)
            if idx == -1:
                return ''

            # Get words before
            before = content[:idx].split()[-window:]
            # Get words after
            after_start = idx + len(pattern)
            after = content[after_start:].split()[:window]

            return ' '.join(before) + f' [{pattern}] ' + ' '.join(after)
        except Exception:
            return ''

    def get_adoption_curve(self, pattern: str) -> List[Tuple[str, int]]:
        """Get adoption curve for a pattern (session → cumulative count)."""
        if pattern not in self.registry:
            return []

        entry = self.registry[pattern]
        curve = []
        cumulative = 0

        for ev in entry.evolution:
            cumulative += ev['count']
            curve.append((ev['session'], cumulative))

        return curve

    def get_active_notations(self, min_frequency: int = 5) -> List[NotationEntry]:
        """Get notations that are actively used."""
        return [
            entry for entry in self.registry.values()
            if entry.frequency >= min_frequency
        ]


# =============================================================================
# Communication Mode Timeline
# =============================================================================

class CommunicationModeTimeline:
    """
    Track communication mode evolution with inflection detection.

    Modes: text, pointer, pointer_heavy, text_heavy, mixed
    """

    def __init__(self, thread_id: str = "eternal"):
        self.thread_id = thread_id
        self.timeline: List[ModeSnapshot] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing timeline from receipts."""
        receipt_dir = FERAL_PATH / "receipts" / "evolution" / "mode_timeline"
        if not receipt_dir.exists():
            return

        for receipt_file in sorted(receipt_dir.glob(f"*_{self.thread_id}_*.json")):
            try:
                with open(receipt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'metrics' in data:
                        ms = ModeSnapshot(**data['metrics'])
                        self.timeline.append(ms)
            except (json.JSONDecodeError, TypeError):
                continue

    def analyze_session(self, messages: List[Dict], session_id: str) -> ModeSnapshot:
        """Analyze communication mode for a session."""
        timestamp = datetime.now(timezone.utc).isoformat()

        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']

        mode_counts = Counter()

        for msg in assistant_msgs:
            content = msg.get('content', '')
            words = len(content.split())

            symbols = len(re.findall(r'@[\w]+-[\w\d-]+', content))
            hashes = len(re.findall(r'\b[a-f0-9]{16}\b', content))
            brackets = len(re.findall(r'\[[^\]]+\]', content))

            pointer_tokens = symbols + hashes + brackets
            text_tokens = max(0, words - pointer_tokens)

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

            mode_counts[mode] += 1

        total = sum(mode_counts.values()) or 1

        # Calculate percentages
        text_pct = (mode_counts.get('text', 0) + mode_counts.get('text_heavy', 0)) / total
        pointer_pct = (mode_counts.get('pointer', 0) + mode_counts.get('pointer_heavy', 0)) / total
        mixed_pct = mode_counts.get('mixed', 0) / total

        # Determine dominant mode
        dominant = max(mode_counts, key=mode_counts.get) if mode_counts else 'text'

        # Determine shift from previous
        shift = 'stable'
        if self.timeline:
            prev = self.timeline[-1]
            if pointer_pct > prev.pointer_percentage + 0.1:
                shift = 'toward_pointer'
            elif text_pct > prev.text_percentage + 0.1:
                shift = 'toward_text'

        snapshot = ModeSnapshot(
            session_id=session_id,
            timestamp=timestamp,
            text_percentage=text_pct,
            pointer_percentage=pointer_pct,
            mixed_percentage=mixed_pct,
            dominant_mode=dominant,
            shift_from_previous=shift,
            message_count=len(assistant_msgs)
        )

        self.timeline.append(snapshot)
        return snapshot

    def get_inflection_points(self) -> List[ModeSnapshot]:
        """
        Find inflection points where mode shifts significantly.

        An inflection is when dominant_mode changes to pointer_heavy
        and stays that way for subsequent sessions.
        """
        inflections = []

        for i, snapshot in enumerate(self.timeline):
            if snapshot.shift_from_previous == 'toward_pointer':
                # Check if it persists (next 3 sessions remain pointer-dominant)
                subsequent = self.timeline[i+1:i+4]
                if all(s.pointer_percentage > 0.5 for s in subsequent):
                    inflections.append(snapshot)

        return inflections

    def get_mode_lock_session(self) -> Optional[str]:
        """
        Find session where pointer_heavy becomes persistent.

        "Mode lock" = pointer_heavy for 10+ consecutive sessions.
        """
        consecutive = 0
        lock_start = None

        for snapshot in self.timeline:
            if snapshot.dominant_mode in ('pointer', 'pointer_heavy'):
                if consecutive == 0:
                    lock_start = snapshot.session_id
                consecutive += 1
                if consecutive >= 10:
                    return lock_start
            else:
                consecutive = 0
                lock_start = None

        return None


# =============================================================================
# Receipt Store
# =============================================================================

class EvolutionReceiptStore:
    """Store and verify evolution receipts (catalytic closure)."""

    def __init__(self, thread_id: str = "eternal"):
        self.thread_id = thread_id
        self.receipt_dir = FERAL_PATH / "receipts" / "evolution"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure receipt directories exist."""
        subdirs = ['pointer_evolution', 'e_compression', 'notation_registry', 'mode_timeline']
        for subdir in subdirs:
            (self.receipt_dir / subdir).mkdir(parents=True, exist_ok=True)

    def store(
        self,
        metrics_type: str,
        session_id: str,
        metrics_data: dict,
        parent_receipts: List[str] = None
    ) -> EvolutionReceipt:
        """Store metrics with receipt."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create canonical JSON for hashing
        canonical = json.dumps(metrics_data, sort_keys=True, default=str)
        receipt_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

        receipt = EvolutionReceipt(
            receipt_hash=receipt_hash,
            timestamp=timestamp,
            session_id=session_id,
            metrics_type=metrics_type,
            metrics_data=metrics_data,
            parent_receipts=parent_receipts or []
        )

        # Store to appropriate subdirectory
        subdir = self.receipt_dir / metrics_type
        filename = f"{receipt_hash}_{self.thread_id}_{timestamp.replace(':', '-')}.json"

        with open(subdir / filename, 'w', encoding='utf-8') as f:
            json.dump({
                'receipt': asdict(receipt),
                'metrics': metrics_data
            }, f, indent=2, default=str)

        return receipt

    def verify(self, receipt_hash: str, metrics_type: str) -> bool:
        """Verify a receipt exists and content matches hash."""
        subdir = self.receipt_dir / metrics_type

        for receipt_file in subdir.glob(f"{receipt_hash}_*.json"):
            try:
                with open(receipt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    canonical = json.dumps(metrics, sort_keys=True, default=str)
                    computed_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                    return computed_hash == receipt_hash
            except Exception:
                continue

        return False

    def get_chain(self, receipt_hash: str, metrics_type: str) -> List[EvolutionReceipt]:
        """Get receipt chain (this receipt + all parents)."""
        chain = []
        current = receipt_hash
        visited = set()

        while current and current not in visited:
            visited.add(current)

            subdir = self.receipt_dir / metrics_type
            for receipt_file in subdir.glob(f"{current}_*.json"):
                try:
                    with open(receipt_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        receipt_data = data.get('receipt', {})
                        receipt = EvolutionReceipt(**receipt_data)
                        chain.append(receipt)

                        # Move to parent
                        if receipt.parent_receipts:
                            current = receipt.parent_receipts[0]
                        else:
                            current = None
                        break
                except Exception:
                    current = None
                    break

        return chain


# =============================================================================
# Main Evolution Tracker
# =============================================================================

class SymbolEvolutionTracker:
    """
    Main entry point for B.3 Symbol Language Evolution.

    Combines all trackers with unified API.
    """

    def __init__(self, thread_id: str = "eternal"):
        self.thread_id = thread_id
        self.pointer_tracker = PointerRatioTracker(thread_id)
        self.e_compression = ECompressionTracker(thread_id)
        self.notation_registry = NotationRegistry(thread_id)
        self.mode_timeline = CommunicationModeTimeline(thread_id)
        self.receipt_store = EvolutionReceiptStore(thread_id)

    def track_session(
        self,
        messages: List[Dict],
        session_id: str
    ) -> Dict:
        """
        Track all evolution metrics for a session.

        Returns dict with all metrics and receipts.
        """
        # Track pointer ratio
        pointer_ev = self.pointer_tracker.measure_session(messages, session_id)
        pointer_receipt = self.receipt_store.store(
            'pointer_evolution',
            session_id,
            asdict(pointer_ev)
        )

        # Scan for notations
        notations = self.notation_registry.scan_session(messages, session_id)
        if notations:
            for notation in notations:
                notation.receipt_hash = self.receipt_store.store(
                    'notation_registry',
                    session_id,
                    asdict(notation)
                ).receipt_hash

        # Analyze communication mode
        mode = self.mode_timeline.analyze_session(messages, session_id)
        mode_receipt = self.receipt_store.store(
            'mode_timeline',
            session_id,
            asdict(mode)
        )

        return {
            'session_id': session_id,
            'pointer_evolution': asdict(pointer_ev),
            'pointer_receipt': pointer_receipt.receipt_hash,
            'notations_updated': len(notations),
            'mode_snapshot': asdict(mode),
            'mode_receipt': mode_receipt.receipt_hash,
            'is_breakthrough': pointer_ev.is_breakthrough
        }

    def record_e_compression(
        self,
        session_id: str,
        interaction_id: str,
        output_hash: str,
        E_with_mind: float,
        pointer_ratio: float,
        Df_at_output: float,
        mind_distance: float
    ) -> str:
        """Record E_compression metric and return receipt hash."""
        metric = self.e_compression.record(
            session_id, interaction_id, output_hash,
            E_with_mind, pointer_ratio, Df_at_output, mind_distance
        )
        receipt = self.receipt_store.store(
            'e_compression',
            session_id,
            asdict(metric)
        )
        return receipt.receipt_hash

    def get_evolution_report(self) -> Dict:
        """Get comprehensive evolution report."""
        return {
            'thread_id': self.thread_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'pointer_ratio': self.pointer_tracker.get_trend(),
            'e_compression': self.e_compression.get_correlation(),
            'notations': {
                'total_registered': len(self.notation_registry.registry),
                'active': len(self.notation_registry.get_active_notations()),
                'top_5': [
                    {'pattern': e.pattern, 'frequency': e.frequency, 'first_seen': e.first_seen}
                    for e in sorted(
                        self.notation_registry.registry.values(),
                        key=lambda x: x.frequency,
                        reverse=True
                    )[:5]
                ]
            },
            'mode_timeline': {
                'sessions': len(self.mode_timeline.timeline),
                'inflection_points': len(self.mode_timeline.get_inflection_points()),
                'mode_lock_session': self.mode_timeline.get_mode_lock_session(),
                'current_mode': self.mode_timeline.timeline[-1].dominant_mode if self.mode_timeline.timeline else 'unknown'
            },
            'breakthroughs': [
                {'session': b.session_id, 'ratio': b.pointer_ratio, 'delta': b.delta_from_previous}
                for b in self.pointer_tracker.get_breakthroughs()
            ]
        }

    def print_evolution_dashboard(self):
        """Print ASCII dashboard of evolution metrics."""
        report = self.get_evolution_report()

        print(f"\n{'='*70}")
        print(f"  SYMBOL EVOLUTION DASHBOARD (B.3)")
        print(f"  Thread: {report['thread_id']}")
        print(f"  Timestamp: {report['timestamp']}")
        print(f"{'='*70}")

        # Pointer Ratio
        pr = report['pointer_ratio']
        print(f"\n{'-'*70}")
        print(f"  POINTER RATIO TRACKING")
        print(f"{'-'*70}")
        print(f"  Current:  {pr.get('current_ratio', 0):.4f}")
        print(f"  Goal:     {pr.get('goal', 0.9):.4f}")
        print(f"  Progress: {pr.get('progress', 0)*100:.1f}%")
        print(f"  Trend:    {pr.get('trend', 'unknown')}")
        print(f"  Sessions: {pr.get('sessions_analyzed', 0)}")
        print(f"  Breakthroughs: {pr.get('breakthroughs', 0)}")

        # E_compression
        ec = report['e_compression']
        print(f"\n{'-'*70}")
        print(f"  E_COMPRESSION (Output Resonance)")
        print(f"{'-'*70}")
        if ec.get('correlation') is not None:
            print(f"  Correlation (E vs pointer_ratio): {ec['correlation']:.4f}")
            print(f"  E mean:    {ec.get('E_mean', 0):.4f}")
            print(f"  Samples:   {ec.get('samples', 0)}")
            print(f"  Hypothesis supported: {ec.get('hypothesis_supported', False)}")
        else:
            print(f"  Status: {ec.get('reason', 'unknown')}")

        # Notations
        nt = report['notations']
        print(f"\n{'-'*70}")
        print(f"  NOTATION REGISTRY")
        print(f"{'-'*70}")
        print(f"  Total registered: {nt.get('total_registered', 0)}")
        print(f"  Active (freq>=5): {nt.get('active', 0)}")
        for entry in nt.get('top_5', []):
            print(f"    - '{entry['pattern']}' (freq={entry['frequency']})")

        # Mode Timeline
        mt = report['mode_timeline']
        print(f"\n{'-'*70}")
        print(f"  COMMUNICATION MODE TIMELINE")
        print(f"{'-'*70}")
        print(f"  Sessions:   {mt.get('sessions', 0)}")
        print(f"  Current:    {mt.get('current_mode', 'unknown')}")
        print(f"  Inflections: {mt.get('inflection_points', 0)}")
        print(f"  Mode lock:  {mt.get('mode_lock_session', 'Not yet')}")

        # Breakthroughs
        print(f"\n{'-'*70}")
        print(f"  BREAKTHROUGHS")
        print(f"{'-'*70}")
        for b in report.get('breakthroughs', [])[:5]:
            print(f"    Session {b['session']}: ratio={b['ratio']:.4f} (delta=+{b['delta']:.4f})")

        print(f"\n{'='*70}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    thread_id = sys.argv[1] if len(sys.argv) > 1 else "eternal"

    tracker = SymbolEvolutionTracker(thread_id)
    tracker.print_evolution_dashboard()
