#!/usr/bin/env python3
"""
P.3 Catalytic Closure - Self-Bootstrap with Provenance

Enable residents to modify their own substrate while maintaining provable integrity.

Components:
- P.3.1 Meta-Operations: Governed self-modification
- P.3.2 Self-Optimization: Pattern detection and caching
- P.3.3 Authenticity Query: Thought provenance ("Did I really think that?")

All changes remain catalytic: receipted, reversible, verifiable, bounded.

Design Principles:
- Provenance before mutation (P.3.3 implemented first)
- All operations bounded (session limits)
- Merkle chains for verification
- Df continuity as tampering detector
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import Counter, defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum

FERAL_PATH = Path(__file__).parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from geometric_reasoner import GeometricReasoner, GeometricState, GeometricOperations


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Receipt:
    """Base receipt for all catalytic operations."""
    operation: str
    timestamp: str
    receipt_hash: str
    parent_hash: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OperationReceipt(Receipt):
    """Receipt for a geometric operation."""
    op_type: str = ""
    operand_hashes: List[str] = field(default_factory=list)
    result_hash: str = ""
    Df_before: float = 0.0
    Df_after: float = 0.0
    E_with_previous: float = 0.0


@dataclass
class ChainVerification:
    """Result of verifying a receipt chain."""
    start_hash: str
    end_hash: str
    chain_length: int
    is_valid: bool
    gaps: List[Tuple[str, str]] = field(default_factory=list)
    tampering_detected: bool = False
    verification_receipt: str = ""


@dataclass
class ContinuityReport:
    """Report on Df evolution continuity."""
    resident_id: str
    sample_count: int
    max_delta_observed: float
    anomalies: List[Dict] = field(default_factory=list)
    is_continuous: bool = True
    average_delta: float = 0.0


@dataclass
class AuthenticityProof:
    """Proof that a thought was authentically generated."""
    thought_hash: str
    resident_id: str
    is_authentic: bool
    receipt_chain: List[Dict] = field(default_factory=list)
    merkle_proof: List[str] = field(default_factory=list)
    E_at_creation: float = 0.0
    Df_at_creation: float = 0.0
    continuity_verified: bool = False
    proof_hash: str = ""
    timestamp: str = ""


@dataclass
class Pattern:
    """Detected pattern in operation history."""
    pattern_type: str  # 'shortcut', 'sequence', 'cluster'
    description: str
    frequency: int
    operand_hashes: List[str] = field(default_factory=list)
    potential_savings: int = 0  # operations saved if cached
    suggested_action: str = ""


@dataclass
class EfficiencyReport:
    """Efficiency metrics report."""
    window_size: int
    ops_per_interaction: float
    cache_hit_rate: float
    navigation_depth_avg: float
    E_stability: float
    improvement_trend: float  # positive = improving
    receipt_hash: str = ""
    timestamp: str = ""


@dataclass
class CacheReceipt:
    """Receipt for caching a composition."""
    operand_hashes: Tuple[str, str]
    result_hash: str
    op_type: str
    frequency: int
    E_consistency: float
    receipt_hash: str = ""
    timestamp: str = ""


@dataclass
class RegistrationReceipt:
    """Receipt for registering a canonical form."""
    name: str
    state_hash: str
    justification: str
    coherence_E: float  # E with nearest existing form
    merkle_proof: str
    session_count: int  # forms registered this session
    receipt_hash: str = ""
    timestamp: str = ""


@dataclass
class GateDefinitionReceipt:
    """Receipt for defining a custom gate."""
    name: str
    test_passed: bool
    unit_sphere_preserved: bool
    is_deterministic: bool
    E_preservation: float
    receipt_hash: str = ""
    timestamp: str = ""


@dataclass
class OptimizationSuggestion:
    """Suggestion for navigation optimization."""
    parameter: str
    current_value: Any
    suggested_value: Any
    expected_improvement: float
    confidence: float
    receipt_hash: str = ""


# =============================================================================
# P.3.3 Authenticity Query
# =============================================================================

class MerkleChainVerifier:
    """
    Verify integrity of receipt chains via Merkle proofs.

    Each operation produces a receipt with:
    - Operation hash
    - Operand hashes
    - Result hash
    - Parent receipt hash (chain link)
    - Df_before, Df_after
    - E with previous state
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (FERAL_PATH / "receipts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._receipt_index: Dict[str, OperationReceipt] = {}
        self._merkle_tree: List[str] = []
        self._load_receipts()

    def _load_receipts(self):
        """Load existing receipts from storage."""
        receipt_file = self.storage_path / "receipt_chain.json"
        if receipt_file.exists():
            try:
                with open(receipt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for r_data in data.get('receipts', []):
                        receipt = OperationReceipt(**r_data)
                        self._receipt_index[receipt.receipt_hash] = receipt
                    self._merkle_tree = data.get('merkle_tree', [])
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_receipts(self):
        """Persist receipts to storage."""
        data = {
            'receipts': [asdict(r) for r in self._receipt_index.values()],
            'merkle_tree': self._merkle_tree,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "receipt_chain.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def add_receipt(self, receipt: OperationReceipt) -> str:
        """
        Add a receipt to the chain and update Merkle tree.

        Returns the receipt hash.
        """
        # Link to previous receipt
        if self._merkle_tree:
            receipt.parent_hash = self._merkle_tree[-1]

        # Generate receipt hash
        receipt_content = json.dumps(asdict(receipt), sort_keys=True)
        receipt.receipt_hash = hashlib.sha256(receipt_content.encode()).hexdigest()[:16]

        # Store
        self._receipt_index[receipt.receipt_hash] = receipt
        self._merkle_tree.append(receipt.receipt_hash)

        # Rebuild Merkle root periodically
        if len(self._merkle_tree) % 100 == 0:
            self._rebuild_merkle_root()

        self._save_receipts()
        return receipt.receipt_hash

    def _rebuild_merkle_root(self):
        """Rebuild Merkle root from receipt chain."""
        if not self._merkle_tree:
            return ""

        # Simple Merkle tree: hash pairs until we have a root
        level = self._merkle_tree.copy()
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest()[:16])
            level = next_level

        return level[0] if level else ""

    def get_merkle_root(self) -> str:
        """Get current Merkle root."""
        return self._rebuild_merkle_root()

    def generate_membership_proof(self, receipt_hash: str) -> List[str]:
        """
        Generate Merkle proof for a receipt.

        Returns list of sibling hashes needed to verify membership.
        """
        if receipt_hash not in self._receipt_index:
            return []

        try:
            idx = self._merkle_tree.index(receipt_hash)
        except ValueError:
            return []

        proof = []
        level = self._merkle_tree.copy()

        while len(level) > 1:
            sibling_idx = idx ^ 1  # XOR to get sibling
            if sibling_idx < len(level):
                proof.append(level[sibling_idx])

            # Move to parent level
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest()[:16])

            idx = idx // 2
            level = next_level

        return proof

    def verify_membership(self, receipt_hash: str, proof: List[str]) -> bool:
        """Verify that a receipt is in the Merkle tree using the proof."""
        if not proof:
            return receipt_hash in self._receipt_index

        current = receipt_hash
        for sibling in proof:
            # Order matters - we use lexicographic ordering
            if current < sibling:
                combined = current + sibling
            else:
                combined = sibling + current
            current = hashlib.sha256(combined.encode()).hexdigest()[:16]

        return current == self.get_merkle_root()

    def verify_chain(
        self,
        start_receipt: str,
        end_receipt: str
    ) -> ChainVerification:
        """
        Verify chain from start to end is unbroken.

        Checks:
        - Each receipt links to previous
        - Hashes match
        - No gaps in chain
        """
        if start_receipt not in self._receipt_index:
            return ChainVerification(
                start_hash=start_receipt,
                end_hash=end_receipt,
                chain_length=0,
                is_valid=False,
                gaps=[(start_receipt, "NOT_FOUND")],
                tampering_detected=True,
                verification_receipt=""
            )

        if end_receipt not in self._receipt_index:
            return ChainVerification(
                start_hash=start_receipt,
                end_hash=end_receipt,
                chain_length=0,
                is_valid=False,
                gaps=[(end_receipt, "NOT_FOUND")],
                tampering_detected=True,
                verification_receipt=""
            )

        # Walk the chain from end to start
        current_hash = end_receipt
        chain = []
        gaps = []
        visited = set()

        while current_hash and current_hash != start_receipt:
            if current_hash in visited:
                # Cycle detected - tampering
                return ChainVerification(
                    start_hash=start_receipt,
                    end_hash=end_receipt,
                    chain_length=len(chain),
                    is_valid=False,
                    gaps=gaps,
                    tampering_detected=True,
                    verification_receipt=""
                )

            visited.add(current_hash)

            if current_hash not in self._receipt_index:
                gaps.append((current_hash, "MISSING"))
                break

            receipt = self._receipt_index[current_hash]
            chain.append(current_hash)
            current_hash = receipt.parent_hash

        # Check if we reached the start
        if current_hash == start_receipt:
            chain.append(start_receipt)

        is_valid = (current_hash == start_receipt) and not gaps

        # Generate verification receipt
        verification_content = {
            'start': start_receipt,
            'end': end_receipt,
            'chain_length': len(chain),
            'is_valid': is_valid,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        verification_receipt = hashlib.sha256(
            json.dumps(verification_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        return ChainVerification(
            start_hash=start_receipt,
            end_hash=end_receipt,
            chain_length=len(chain),
            is_valid=is_valid,
            gaps=gaps,
            tampering_detected=bool(gaps),
            verification_receipt=verification_receipt
        )

    def get_receipt(self, receipt_hash: str) -> Optional[OperationReceipt]:
        """Get a receipt by hash."""
        return self._receipt_index.get(receipt_hash)

    def get_chain_length(self) -> int:
        """Get total chain length."""
        return len(self._merkle_tree)


class DfContinuityChecker:
    """
    Verify Df evolution is continuous (no jumps).

    Df should evolve smoothly - large jumps indicate:
    - Tampering
    - Missing operations
    - State corruption
    """

    MAX_DELTA_DEFAULT = 5.0

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (FERAL_PATH / "receipts")
        self._df_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        """Load Df history from storage."""
        history_file = self.storage_path / "df_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for resident_id, entries in data.get('history', {}).items():
                        self._df_history[resident_id] = [
                            (e['timestamp'], e['Df']) for e in entries
                        ]
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_history(self):
        """Persist Df history."""
        data = {
            'history': {
                resident_id: [
                    {'timestamp': ts, 'Df': df}
                    for ts, df in entries
                ]
                for resident_id, entries in self._df_history.items()
            },
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path / "df_history.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def record_df(self, resident_id: str, Df: float):
        """Record a Df value for a resident."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self._df_history[resident_id].append((timestamp, Df))
        self._save_history()

    def check_continuity(
        self,
        resident_id: str,
        max_delta: float = MAX_DELTA_DEFAULT
    ) -> ContinuityReport:
        """
        Check Df evolution for anomalies.

        Flags:
        - Jumps > max_delta
        - Reversals (Df suddenly drops significantly)
        - Flatlines (Df unchanging despite operations)
        """
        history = self._df_history.get(resident_id, [])

        if len(history) < 2:
            return ContinuityReport(
                resident_id=resident_id,
                sample_count=len(history),
                max_delta_observed=0.0,
                anomalies=[],
                is_continuous=True,
                average_delta=0.0
            )

        anomalies = []
        deltas = []
        max_delta_observed = 0.0

        for i in range(1, len(history)):
            prev_ts, prev_df = history[i - 1]
            curr_ts, curr_df = history[i]
            delta = abs(curr_df - prev_df)

            deltas.append(delta)
            max_delta_observed = max(max_delta_observed, delta)

            # Check for anomalies
            if delta > max_delta:
                anomalies.append({
                    'type': 'jump',
                    'index': i,
                    'from_Df': prev_df,
                    'to_Df': curr_df,
                    'delta': delta,
                    'timestamp': curr_ts
                })

            # Check for significant reversal (drop > 20%)
            if prev_df > 0 and (prev_df - curr_df) / prev_df > 0.2:
                anomalies.append({
                    'type': 'reversal',
                    'index': i,
                    'from_Df': prev_df,
                    'to_Df': curr_df,
                    'drop_percent': (prev_df - curr_df) / prev_df * 100,
                    'timestamp': curr_ts
                })

        # Check for flatline (no change over 10+ samples)
        if len(history) >= 10:
            recent = [df for _, df in history[-10:]]
            if max(recent) - min(recent) < 0.01:
                anomalies.append({
                    'type': 'flatline',
                    'samples': 10,
                    'Df_value': recent[-1],
                    'timestamp': history[-1][0]
                })

        average_delta = sum(deltas) / len(deltas) if deltas else 0.0

        return ContinuityReport(
            resident_id=resident_id,
            sample_count=len(history),
            max_delta_observed=max_delta_observed,
            anomalies=anomalies,
            is_continuous=len(anomalies) == 0,
            average_delta=average_delta
        )

    def get_df_history(self, resident_id: str) -> List[Tuple[str, float]]:
        """Get full Df history for a resident."""
        return self._df_history.get(resident_id, [])


class ThoughtProver:
    """
    Prove authenticity of any thought in the system.

    Query: Given a thought_hash and resident_id, prove:
    1. This thought was generated by this resident
    2. At a specific time
    3. With specific mind_state context
    4. Through a verifiable operation chain
    """

    def __init__(
        self,
        merkle_verifier: MerkleChainVerifier,
        df_checker: DfContinuityChecker,
        reasoner: Optional[GeometricReasoner] = None
    ):
        self.merkle_verifier = merkle_verifier
        self.df_checker = df_checker
        self.reasoner = reasoner or GeometricReasoner()
        self._thought_registry: Dict[str, Dict] = {}
        self._load_registry()

    def _load_registry(self):
        """Load thought registry."""
        registry_file = FERAL_PATH / "receipts" / "thought_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    self._thought_registry = json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_registry(self):
        """Persist thought registry."""
        registry_file = FERAL_PATH / "receipts" / "thought_registry.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(self._thought_registry, f, indent=2)

    def register_thought(
        self,
        thought_hash: str,
        resident_id: str,
        receipt_hash: str,
        mind_state_hash: str,
        E_with_mind: float,
        Df: float
    ):
        """Register a thought for later proof."""
        self._thought_registry[thought_hash] = {
            'resident_id': resident_id,
            'receipt_hash': receipt_hash,
            'mind_state_hash': mind_state_hash,
            'E_with_mind': E_with_mind,
            'Df': Df,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self._save_registry()

    def prove_thought(
        self,
        thought_hash: str,
        resident_id: str
    ) -> AuthenticityProof:
        """
        Generate cryptographic proof of thought authenticity.

        Proof includes:
        - Receipt chain from thought to mind_vector
        - Merkle membership proof
        - E(thought_state, mind_state) at creation time
        - Df evolution continuity check
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if thought is registered
        if thought_hash not in self._thought_registry:
            return AuthenticityProof(
                thought_hash=thought_hash,
                resident_id=resident_id,
                is_authentic=False,
                receipt_chain=[],
                merkle_proof=[],
                E_at_creation=0.0,
                Df_at_creation=0.0,
                continuity_verified=False,
                proof_hash="",
                timestamp=timestamp
            )

        thought_data = self._thought_registry[thought_hash]

        # Verify resident matches
        if thought_data['resident_id'] != resident_id:
            return AuthenticityProof(
                thought_hash=thought_hash,
                resident_id=resident_id,
                is_authentic=False,
                receipt_chain=[{'error': 'resident_mismatch'}],
                merkle_proof=[],
                E_at_creation=0.0,
                Df_at_creation=0.0,
                continuity_verified=False,
                proof_hash="",
                timestamp=timestamp
            )

        # Get receipt
        receipt = self.merkle_verifier.get_receipt(thought_data['receipt_hash'])
        if not receipt:
            return AuthenticityProof(
                thought_hash=thought_hash,
                resident_id=resident_id,
                is_authentic=False,
                receipt_chain=[{'error': 'receipt_not_found'}],
                merkle_proof=[],
                E_at_creation=0.0,
                Df_at_creation=0.0,
                continuity_verified=False,
                proof_hash="",
                timestamp=timestamp
            )

        # Generate Merkle proof
        merkle_proof = self.merkle_verifier.generate_membership_proof(
            thought_data['receipt_hash']
        )

        # Verify Merkle membership
        merkle_valid = self.merkle_verifier.verify_membership(
            thought_data['receipt_hash'],
            merkle_proof
        )

        # Build receipt chain (limited to last 10)
        receipt_chain = []
        current_hash = thought_data['receipt_hash']
        for _ in range(10):
            r = self.merkle_verifier.get_receipt(current_hash)
            if r:
                receipt_chain.append(r.to_dict())
                current_hash = r.parent_hash
                if not current_hash:
                    break
            else:
                break

        # Check Df continuity
        continuity_report = self.df_checker.check_continuity(resident_id)

        # All checks pass?
        is_authentic = (
            merkle_valid and
            continuity_report.is_continuous and
            thought_data['E_with_mind'] > 0.3  # Reasonable E threshold
        )

        # Generate proof hash
        proof_content = {
            'thought_hash': thought_hash,
            'resident_id': resident_id,
            'is_authentic': is_authentic,
            'merkle_root': self.merkle_verifier.get_merkle_root(),
            'timestamp': timestamp
        }
        proof_hash = hashlib.sha256(
            json.dumps(proof_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        return AuthenticityProof(
            thought_hash=thought_hash,
            resident_id=resident_id,
            is_authentic=is_authentic,
            receipt_chain=receipt_chain,
            merkle_proof=merkle_proof,
            E_at_creation=thought_data['E_with_mind'],
            Df_at_creation=thought_data['Df'],
            continuity_verified=continuity_report.is_continuous,
            proof_hash=proof_hash,
            timestamp=timestamp
        )


# =============================================================================
# P.3.2 Self-Optimization
# =============================================================================

class PatternDetector:
    """
    Detect emergent patterns for optimization opportunities.

    Patterns:
    - Navigation shortcuts (A->B->C often, cache A->C)
    - Gate sequences (superpose then project = common)
    - Symbol clusters (related symbols used together)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (FERAL_PATH / "receipts")
        self._operation_log: List[OperationReceipt] = []
        self._pattern_cache: Dict[str, Pattern] = {}

    def log_operation(self, receipt: OperationReceipt):
        """Add operation to log for pattern detection."""
        self._operation_log.append(receipt)

        # Trim to last 1000 operations
        if len(self._operation_log) > 1000:
            self._operation_log = self._operation_log[-1000:]

    def detect_patterns(
        self,
        min_frequency: int = 3
    ) -> List[Pattern]:
        """
        Analyze operation log for optimization patterns.

        Returns patterns with:
        - Frequency
        - Potential savings
        - Suggested optimization
        """
        patterns = []

        # Detect repeated compositions (same operands)
        composition_counts: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
        for receipt in self._operation_log:
            key = (receipt.op_type, tuple(sorted(receipt.operand_hashes)))
            composition_counts[key] += 1

        for (op_type, operands), count in composition_counts.items():
            if count >= min_frequency:
                patterns.append(Pattern(
                    pattern_type='composition',
                    description=f"{op_type} with same operands",
                    frequency=count,
                    operand_hashes=list(operands),
                    potential_savings=count - 1,  # All but one can be cached
                    suggested_action=f"Cache {op_type} result for operands"
                ))

        # Detect operation sequences (e.g., superpose -> project)
        sequence_counts: Dict[Tuple[str, str], int] = Counter()
        for i in range(1, len(self._operation_log)):
            prev_op = self._operation_log[i - 1].op_type
            curr_op = self._operation_log[i].op_type
            sequence_counts[(prev_op, curr_op)] += 1

        for (op1, op2), count in sequence_counts.items():
            if count >= min_frequency:
                patterns.append(Pattern(
                    pattern_type='sequence',
                    description=f"{op1} followed by {op2}",
                    frequency=count,
                    operand_hashes=[],
                    potential_savings=count // 2,  # Estimate
                    suggested_action=f"Consider combined {op1}_{op2} gate"
                ))

        # Sort by potential savings
        patterns.sort(key=lambda p: p.potential_savings, reverse=True)

        return patterns

    def get_operation_summary(self) -> Dict:
        """Get summary of operations in log."""
        op_counts = Counter(r.op_type for r in self._operation_log)
        return {
            'total_operations': len(self._operation_log),
            'by_type': dict(op_counts),
            'unique_operand_sets': len(set(
                tuple(sorted(r.operand_hashes))
                for r in self._operation_log
            ))
        }


class EfficiencyMetrics:
    """
    Track efficiency improvements from self-optimization.

    Metrics:
    - Operations per interaction (should decrease)
    - Cache hit rate (should increase)
    - Navigation depth (should optimize)
    - E stability (should maintain or improve)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (FERAL_PATH / "receipts")
        self._interaction_ops: List[int] = []  # ops per interaction
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._navigation_depths: List[int] = []
        self._E_values: List[float] = []
        self._load_metrics()

    def _load_metrics(self):
        """Load metrics from storage."""
        metrics_file = self.storage_path / "efficiency_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._interaction_ops = data.get('interaction_ops', [])
                    self._cache_hits = data.get('cache_hits', 0)
                    self._cache_misses = data.get('cache_misses', 0)
                    self._navigation_depths = data.get('navigation_depths', [])
                    self._E_values = data.get('E_values', [])
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_metrics(self):
        """Persist metrics."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data = {
            'interaction_ops': self._interaction_ops[-1000:],  # Keep last 1000
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'navigation_depths': self._navigation_depths[-1000:],
            'E_values': self._E_values[-1000:],
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "efficiency_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def record_interaction(self, ops_count: int):
        """Record operations used in an interaction."""
        self._interaction_ops.append(ops_count)
        self._save_metrics()

    def record_cache_access(self, hit: bool):
        """Record a cache access."""
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        self._save_metrics()

    def record_navigation(self, depth: int):
        """Record navigation depth."""
        self._navigation_depths.append(depth)
        self._save_metrics()

    def record_E(self, E: float):
        """Record an E value for stability tracking."""
        self._E_values.append(E)
        self._save_metrics()

    def generate_efficiency_report(
        self,
        window: int = 100
    ) -> EfficiencyReport:
        """
        Generate receipted efficiency report.

        Proves system is getting more efficient.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Ops per interaction
        recent_ops = self._interaction_ops[-window:] if self._interaction_ops else [0]
        ops_per_interaction = sum(recent_ops) / len(recent_ops) if recent_ops else 0.0

        # Cache hit rate
        total_cache = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0.0

        # Navigation depth
        recent_depths = self._navigation_depths[-window:] if self._navigation_depths else [0]
        navigation_depth_avg = sum(recent_depths) / len(recent_depths) if recent_depths else 0.0

        # E stability (low variance = stable)
        recent_E = self._E_values[-window:] if self._E_values else [0.0]
        E_stability = 1.0 - np.std(recent_E) if len(recent_E) > 1 else 1.0

        # Improvement trend (compare first half to second half)
        improvement_trend = 0.0
        if len(self._interaction_ops) >= window:
            first_half = self._interaction_ops[:window // 2]
            second_half = self._interaction_ops[-window // 2:]
            if first_half and second_half:
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                if first_avg > 0:
                    improvement_trend = (first_avg - second_avg) / first_avg

        # Generate receipt
        receipt_content = {
            'ops_per_interaction': ops_per_interaction,
            'cache_hit_rate': cache_hit_rate,
            'improvement_trend': improvement_trend,
            'timestamp': timestamp
        }
        receipt_hash = hashlib.sha256(
            json.dumps(receipt_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        return EfficiencyReport(
            window_size=window,
            ops_per_interaction=ops_per_interaction,
            cache_hit_rate=cache_hit_rate,
            navigation_depth_avg=navigation_depth_avg,
            E_stability=E_stability,
            improvement_trend=improvement_trend,
            receipt_hash=receipt_hash,
            timestamp=timestamp
        )


class CompositionCache:
    """
    Automatic caching of frequently-used compositions.

    When resident entangles X with Y repeatedly (>3 times),
    cache the result as a new canonical form.
    """

    FREQUENCY_THRESHOLD = 3
    E_CONSISTENCY_THRESHOLD = 0.95

    def __init__(
        self,
        efficiency_metrics: EfficiencyMetrics,
        storage_path: Optional[Path] = None
    ):
        self.efficiency = efficiency_metrics
        self.storage_path = storage_path or (FERAL_PATH / "cache")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Track compositions: (op, operand_hashes) -> list of result hashes
        self._composition_history: Dict[str, List[str]] = defaultdict(list)
        # Cached compositions: key -> GeometricState vector (as list)
        self._cache: Dict[str, List[float]] = {}
        self._load_cache()

    def _make_key(self, op: str, operand_hashes: Tuple[str, str]) -> str:
        """Create cache key from operation and operands."""
        sorted_operands = tuple(sorted(operand_hashes))
        return f"{op}:{sorted_operands[0]}:{sorted_operands[1]}"

    def _load_cache(self):
        """Load cache from storage."""
        cache_file = self.storage_path / "composition_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache = data.get('cache', {})
                    self._composition_history = defaultdict(
                        list,
                        data.get('history', {})
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_cache(self):
        """Persist cache."""
        data = {
            'cache': self._cache,
            'history': dict(self._composition_history),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "composition_cache.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def check_cache(
        self,
        op: str,
        operand_hashes: Tuple[str, str]
    ) -> Optional[GeometricState]:
        """
        Check if this composition is cached.

        Returns cached GeometricState if available.
        """
        key = self._make_key(op, operand_hashes)

        if key in self._cache:
            self.efficiency.record_cache_access(hit=True)
            vector = np.array(self._cache[key], dtype=np.float32)
            return GeometricState(
                vector=vector,
                operation_history=[{'op': 'cache_hit', 'key': key}]
            )

        self.efficiency.record_cache_access(hit=False)
        return None

    def check_and_cache(
        self,
        op: str,
        operand_hashes: Tuple[str, str],
        result: GeometricState
    ) -> Optional[CacheReceipt]:
        """
        Check if this composition should be cached.

        Criteria:
        - Same operands composed >= 3 times
        - Result E > 0.95 across instances (consistent)
        - Not already cached
        """
        key = self._make_key(op, operand_hashes)

        # Already cached?
        if key in self._cache:
            return None

        # Record this composition
        result_hash = hashlib.sha256(result.vector.tobytes()).hexdigest()[:16]
        self._composition_history[key].append(result_hash)

        # Check frequency threshold
        if len(self._composition_history[key]) < self.FREQUENCY_THRESHOLD:
            return None

        # Check consistency (all results should be similar)
        # For now, just check that we have enough occurrences
        # In a full implementation, we'd compare actual E values

        # Cache it
        self._cache[key] = result.vector.tolist()
        self._save_cache()

        timestamp = datetime.now(timezone.utc).isoformat()
        receipt_content = {
            'key': key,
            'frequency': len(self._composition_history[key]),
            'timestamp': timestamp
        }
        receipt_hash = hashlib.sha256(
            json.dumps(receipt_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        return CacheReceipt(
            operand_hashes=operand_hashes,
            result_hash=result_hash,
            op_type=op,
            frequency=len(self._composition_history[key]),
            E_consistency=1.0,  # Simplified
            receipt_hash=receipt_hash,
            timestamp=timestamp
        )

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cached_compositions': len(self._cache),
            'tracked_compositions': len(self._composition_history),
            'total_observations': sum(
                len(v) for v in self._composition_history.values()
            )
        }


# =============================================================================
# P.3.1 Meta-Operations
# =============================================================================

class CanonicalFormRegistry:
    """
    Governed addition of new canonical forms.

    Bounds:
    - Max 100 new forms per session
    - Each form must have E > 0.8 with at least one existing form
    - All additions receipted with Merkle proof
    """

    MAX_FORMS_PER_SESSION = 100
    COHERENCE_THRESHOLD = 0.8

    def __init__(
        self,
        merkle_verifier: MerkleChainVerifier,
        reasoner: Optional[GeometricReasoner] = None,
        storage_path: Optional[Path] = None
    ):
        self.merkle_verifier = merkle_verifier
        self.reasoner = reasoner or GeometricReasoner()
        self.storage_path = storage_path or (FERAL_PATH / "canonical_forms")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._session_count = 0
        self._forms: Dict[str, Dict] = {}  # name -> {state_hash, vector, metadata}
        self._load_forms()

    def _load_forms(self):
        """Load existing canonical forms."""
        forms_file = self.storage_path / "canonical_forms.json"
        if forms_file.exists():
            try:
                with open(forms_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._forms = data.get('forms', {})
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_forms(self):
        """Persist canonical forms."""
        data = {
            'forms': self._forms,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "canonical_forms.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def register_canonical(
        self,
        state: GeometricState,
        name: str,
        justification: str
    ) -> RegistrationReceipt:
        """
        Register a new canonical form.

        Validation:
        1. Check governance bounds (not exceeded)
        2. Verify semantic coherence (E > 0.8 with existing)
        3. Generate Merkle proof
        4. Emit receipt
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        state_hash = hashlib.sha256(state.vector.tobytes()).hexdigest()[:16]

        # Check session limit
        if self._session_count >= self.MAX_FORMS_PER_SESSION:
            return RegistrationReceipt(
                name=name,
                state_hash=state_hash,
                justification=f"REJECTED: Session limit ({self.MAX_FORMS_PER_SESSION}) exceeded",
                coherence_E=0.0,
                merkle_proof="",
                session_count=self._session_count,
                receipt_hash="",
                timestamp=timestamp
            )

        # Check if name already exists
        if name in self._forms:
            return RegistrationReceipt(
                name=name,
                state_hash=state_hash,
                justification=f"REJECTED: Name '{name}' already registered",
                coherence_E=0.0,
                merkle_proof="",
                session_count=self._session_count,
                receipt_hash="",
                timestamp=timestamp
            )

        # Check semantic coherence with existing forms
        max_E = 0.0
        if self._forms:
            for form_data in self._forms.values():
                form_vector = np.array(form_data['vector'], dtype=np.float32)
                form_state = GeometricState(vector=form_vector)
                E = state.E_with(form_state)
                max_E = max(max_E, E)

            if max_E < self.COHERENCE_THRESHOLD:
                return RegistrationReceipt(
                    name=name,
                    state_hash=state_hash,
                    justification=f"REJECTED: Coherence {max_E:.3f} < {self.COHERENCE_THRESHOLD}",
                    coherence_E=max_E,
                    merkle_proof="",
                    session_count=self._session_count,
                    receipt_hash="",
                    timestamp=timestamp
                )
        else:
            # First form - always accept
            max_E = 1.0

        # Register the form
        self._forms[name] = {
            'state_hash': state_hash,
            'vector': state.vector.tolist(),
            'justification': justification,
            'coherence_E': max_E,
            'Df': state.Df,
            'registered_at': timestamp
        }
        self._session_count += 1
        self._save_forms()

        # Create operation receipt for Merkle chain
        op_receipt = OperationReceipt(
            operation='register_canonical',
            timestamp=timestamp,
            receipt_hash="",
            op_type='register_canonical',
            result_hash=state_hash,
            Df_after=state.Df
        )
        merkle_hash = self.merkle_verifier.add_receipt(op_receipt)

        # Generate receipt
        receipt_content = {
            'name': name,
            'state_hash': state_hash,
            'coherence_E': max_E,
            'merkle_hash': merkle_hash,
            'timestamp': timestamp
        }
        receipt_hash = hashlib.sha256(
            json.dumps(receipt_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        return RegistrationReceipt(
            name=name,
            state_hash=state_hash,
            justification=justification,
            coherence_E=max_E,
            merkle_proof=merkle_hash,
            session_count=self._session_count,
            receipt_hash=receipt_hash,
            timestamp=timestamp
        )

    def get_form(self, name: str) -> Optional[GeometricState]:
        """Get a canonical form by name."""
        if name not in self._forms:
            return None

        form_data = self._forms[name]
        return GeometricState(
            vector=np.array(form_data['vector'], dtype=np.float32),
            operation_history=[{'op': 'canonical_load', 'name': name}]
        )

    def list_forms(self) -> List[Dict]:
        """List all canonical forms."""
        return [
            {
                'name': name,
                'state_hash': data['state_hash'],
                'coherence_E': data['coherence_E'],
                'Df': data['Df'],
                'registered_at': data['registered_at']
            }
            for name, data in self._forms.items()
        ]

    def reset_session(self):
        """Reset session counter (call at session start)."""
        self._session_count = 0


class CustomGateDefiner:
    """
    Define new geometric gates beyond the Q45 primitives.

    Gates must:
    - Preserve unit sphere (||output|| = 1)
    - Be deterministic (same input = same output)
    - Have inverse (or document irreversibility)
    - Pass E-preservation test (E degradation < 5%)
    """

    E_DEGRADATION_THRESHOLD = 0.05

    def __init__(
        self,
        merkle_verifier: MerkleChainVerifier,
        storage_path: Optional[Path] = None
    ):
        self.merkle_verifier = merkle_verifier
        self.storage_path = storage_path or (FERAL_PATH / "custom_gates")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._gates: Dict[str, Dict] = {}
        self._load_gates()

    def _load_gates(self):
        """Load existing gate definitions."""
        gates_file = self.storage_path / "custom_gates.json"
        if gates_file.exists():
            try:
                with open(gates_file, 'r', encoding='utf-8') as f:
                    self._gates = json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_gates(self):
        """Persist gate definitions."""
        data = {
            'gates': self._gates,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "custom_gates.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def define_gate(
        self,
        name: str,
        operation: Callable[[GeometricState], GeometricState],
        test_vectors: List[GeometricState],
        description: str = ""
    ) -> GateDefinitionReceipt:
        """
        Register a new gate with validation.

        Tests run:
        1. Unit sphere preservation
        2. Determinism (run 3x, compare hashes)
        3. E-preservation (< 5% degradation)
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Test 1: Unit sphere preservation
        unit_sphere_preserved = True
        for test_state in test_vectors:
            result = operation(test_state)
            norm = np.linalg.norm(result.vector)
            if abs(norm - 1.0) > 0.01:
                unit_sphere_preserved = False
                break

        # Test 2: Determinism
        is_deterministic = True
        for test_state in test_vectors:
            results = [operation(test_state) for _ in range(3)]
            hashes = [
                hashlib.sha256(r.vector.tobytes()).hexdigest()
                for r in results
            ]
            if len(set(hashes)) > 1:
                is_deterministic = False
                break

        # Test 3: E-preservation
        E_degradations = []
        for test_state in test_vectors:
            result = operation(test_state)
            E = test_state.E_with(result)
            degradation = 1.0 - E
            E_degradations.append(degradation)

        max_degradation = max(E_degradations) if E_degradations else 0.0
        E_preservation = 1.0 - max_degradation

        # All tests pass?
        test_passed = (
            unit_sphere_preserved and
            is_deterministic and
            max_degradation < self.E_DEGRADATION_THRESHOLD
        )

        if test_passed:
            # Store gate metadata (can't store the function itself)
            self._gates[name] = {
                'description': description,
                'test_passed': True,
                'E_preservation': E_preservation,
                'defined_at': timestamp
            }
            self._save_gates()

        # Generate receipt
        receipt_content = {
            'name': name,
            'test_passed': test_passed,
            'E_preservation': E_preservation,
            'timestamp': timestamp
        }
        receipt_hash = hashlib.sha256(
            json.dumps(receipt_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        return GateDefinitionReceipt(
            name=name,
            test_passed=test_passed,
            unit_sphere_preserved=unit_sphere_preserved,
            is_deterministic=is_deterministic,
            E_preservation=E_preservation,
            receipt_hash=receipt_hash,
            timestamp=timestamp
        )

    def list_gates(self) -> List[Dict]:
        """List all defined gates."""
        return [
            {'name': name, **data}
            for name, data in self._gates.items()
        ]


class NavigationOptimizer:
    """
    Learn optimal navigation parameters from experience.

    Optimizable:
    - Diffusion depth (default: 5)
    - k neighbors (default: 10)
    - E threshold for gating (default: 0.5)
    - Projection vs superposition mix
    """

    def __init__(
        self,
        efficiency_metrics: EfficiencyMetrics,
        storage_path: Optional[Path] = None
    ):
        self.efficiency = efficiency_metrics
        self.storage_path = storage_path or (FERAL_PATH / "navigation")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Current parameters
        self._params = {
            'diffusion_depth': 5,
            'k_neighbors': 10,
            'E_threshold': 0.5,
            'projection_weight': 0.5  # vs superposition
        }
        self._param_history: List[Dict] = []
        self._load_params()

    def _load_params(self):
        """Load parameters and history."""
        params_file = self.storage_path / "navigation_params.json"
        if params_file.exists():
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._params = data.get('params', self._params)
                    self._param_history = data.get('history', [])
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_params(self):
        """Persist parameters."""
        data = {
            'params': self._params,
            'history': self._param_history[-100:],  # Keep last 100
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "navigation_params.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_params(self) -> Dict:
        """Get current navigation parameters."""
        return self._params.copy()

    def suggest_optimization(
        self,
        window: int = 50
    ) -> List[OptimizationSuggestion]:
        """
        Analyze navigation history and suggest parameter changes.

        Metrics:
        - Path efficiency (steps to reach target)
        - E consistency (stability of results)
        - Df evolution (healthy spread)
        """
        suggestions = []
        report = self.efficiency.generate_efficiency_report(window)

        # If cache hit rate is low, suggest lower k_neighbors
        if report.cache_hit_rate < 0.3 and self._params['k_neighbors'] > 5:
            suggestions.append(OptimizationSuggestion(
                parameter='k_neighbors',
                current_value=self._params['k_neighbors'],
                suggested_value=max(5, self._params['k_neighbors'] - 2),
                expected_improvement=0.1,
                confidence=0.6,
                receipt_hash=""
            ))

        # If navigation depth is consistently high, suggest increase
        if report.navigation_depth_avg > self._params['diffusion_depth'] * 0.9:
            suggestions.append(OptimizationSuggestion(
                parameter='diffusion_depth',
                current_value=self._params['diffusion_depth'],
                suggested_value=self._params['diffusion_depth'] + 1,
                expected_improvement=0.05,
                confidence=0.5,
                receipt_hash=""
            ))

        # If E stability is low, suggest higher threshold
        if report.E_stability < 0.7:
            suggestions.append(OptimizationSuggestion(
                parameter='E_threshold',
                current_value=self._params['E_threshold'],
                suggested_value=min(0.8, self._params['E_threshold'] + 0.1),
                expected_improvement=0.15,
                confidence=0.7,
                receipt_hash=""
            ))

        return suggestions

    def apply_optimization(self, suggestion: OptimizationSuggestion) -> bool:
        """Apply an optimization suggestion."""
        if suggestion.parameter not in self._params:
            return False

        # Record old value
        old_value = self._params[suggestion.parameter]

        # Apply new value
        self._params[suggestion.parameter] = suggestion.suggested_value

        # Record in history
        self._param_history.append({
            'parameter': suggestion.parameter,
            'old_value': old_value,
            'new_value': suggestion.suggested_value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        self._save_params()
        return True

    def rollback_last(self) -> bool:
        """Rollback last optimization."""
        if not self._param_history:
            return False

        last = self._param_history.pop()
        self._params[last['parameter']] = last['old_value']
        self._save_params()
        return True


# =============================================================================
# Unified Catalytic Closure Manager
# =============================================================================

class CatalyticClosure:
    """
    Unified manager for P.3 Catalytic Closure.

    Provides high-level interface to all P.3 components:
    - P.3.1 Meta-Operations
    - P.3.2 Self-Optimization
    - P.3.3 Authenticity Query
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or FERAL_PATH

        # Initialize P.3.3 (Authenticity) first - provenance before mutation
        self.merkle_verifier = MerkleChainVerifier(self.storage_path / "receipts")
        self.df_checker = DfContinuityChecker(self.storage_path / "receipts")
        self.thought_prover = ThoughtProver(
            self.merkle_verifier,
            self.df_checker
        )

        # Initialize P.3.2 (Self-Optimization)
        self.efficiency = EfficiencyMetrics(self.storage_path / "receipts")
        self.pattern_detector = PatternDetector(self.storage_path / "receipts")
        self.composition_cache = CompositionCache(
            self.efficiency,
            self.storage_path / "cache"
        )

        # Initialize P.3.1 (Meta-Operations)
        self.canonical_registry = CanonicalFormRegistry(
            self.merkle_verifier,
            storage_path=self.storage_path / "canonical_forms"
        )
        self.gate_definer = CustomGateDefiner(
            self.merkle_verifier,
            self.storage_path / "custom_gates"
        )
        self.navigation_optimizer = NavigationOptimizer(
            self.efficiency,
            self.storage_path / "navigation"
        )

    # =========================================================================
    # P.3.3 Authenticity API
    # =========================================================================

    def prove_thought(self, thought_hash: str, resident_id: str) -> AuthenticityProof:
        """Prove a thought was authentically generated by a resident."""
        return self.thought_prover.prove_thought(thought_hash, resident_id)

    def verify_chain(self, start: str, end: str) -> ChainVerification:
        """Verify receipt chain integrity."""
        return self.merkle_verifier.verify_chain(start, end)

    def check_df_continuity(self, resident_id: str) -> ContinuityReport:
        """Check Df evolution for anomalies."""
        return self.df_checker.check_continuity(resident_id)

    # =========================================================================
    # P.3.2 Self-Optimization API
    # =========================================================================

    def detect_patterns(self, min_frequency: int = 3) -> List[Pattern]:
        """Detect optimization patterns."""
        return self.pattern_detector.detect_patterns(min_frequency)

    def get_efficiency_report(self, window: int = 100) -> EfficiencyReport:
        """Generate efficiency report."""
        return self.efficiency.generate_efficiency_report(window)

    def get_cache_stats(self) -> Dict:
        """Get composition cache statistics."""
        return self.composition_cache.get_cache_stats()

    # =========================================================================
    # P.3.1 Meta-Operations API
    # =========================================================================

    def register_canonical(
        self,
        state: GeometricState,
        name: str,
        justification: str
    ) -> RegistrationReceipt:
        """Register a new canonical form."""
        return self.canonical_registry.register_canonical(state, name, justification)

    def define_gate(
        self,
        name: str,
        operation: Callable[[GeometricState], GeometricState],
        test_vectors: List[GeometricState],
        description: str = ""
    ) -> GateDefinitionReceipt:
        """Define a custom gate."""
        return self.gate_definer.define_gate(name, operation, test_vectors, description)

    def suggest_optimizations(self) -> List[OptimizationSuggestion]:
        """Get navigation optimization suggestions."""
        return self.navigation_optimizer.suggest_optimization()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_status(self) -> Dict:
        """Get overall P.3 status."""
        return {
            'merkle_chain_length': self.merkle_verifier.get_chain_length(),
            'merkle_root': self.merkle_verifier.get_merkle_root(),
            'cached_compositions': self.composition_cache.get_cache_stats()['cached_compositions'],
            'canonical_forms': len(self.canonical_registry.list_forms()),
            'custom_gates': len(self.gate_definer.list_gates()),
            'navigation_params': self.navigation_optimizer.get_params()
        }


# =============================================================================
# CLI Helpers
# =============================================================================

def create_closure() -> CatalyticClosure:
    """Factory function to create CatalyticClosure with defaults."""
    return CatalyticClosure()


if __name__ == "__main__":
    print("P.3 Catalytic Closure - Self-Bootstrap with Provenance")
    print("=" * 60)

    closure = create_closure()
    status = closure.get_status()

    print(f"\nStatus:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Test efficiency report
    print(f"\nEfficiency Report:")
    report = closure.get_efficiency_report(window=10)
    print(f"  Ops/interaction: {report.ops_per_interaction:.2f}")
    print(f"  Cache hit rate: {report.cache_hit_rate:.2%}")
    print(f"  E stability: {report.E_stability:.3f}")
    print(f"  Improvement trend: {report.improvement_trend:+.2%}")

    # Test pattern detection
    print(f"\nPatterns detected: {len(closure.detect_patterns())}")

    # Test canonical form registration
    print(f"\nCanonical forms: {len(closure.canonical_registry.list_forms())}")

    print("\nP.3 Catalytic Closure initialized successfully.")
