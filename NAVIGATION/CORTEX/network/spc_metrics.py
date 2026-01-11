#!/usr/bin/env python3
"""
SPC Metrics - Semantic Density tracking per Q33 derivation.

Implements:
- σ^Df = N (concept_units) measurement
- CDR (Concept Density Ratio) tracking
- ECR (Exact Correctness Rate) tracking
- H(X), H(X|S), I(X;S) measurement

Reference:
- THOUGHT/LAB/FORMULA/research/questions/medium_priority/q33_conditional_entropy_semantic_density.md
- THOUGHT/LAB/FORMULA/research/questions/high_priority/q35_markov_blankets.md
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_concept_units(node: dict) -> int:
    """Count concept_units per GOV_IR_SPEC Section 7.

    Atomic semantic nodes contribute 1 concept_unit each:
    - constraint, permission, prohibition, reference, gate

    Operations combine operand units:
    - AND: sum of operands
    - OR: max of operands
    - NOT: operand value

    Args:
        node: GOV_IR node dictionary

    Returns:
        Integer count of concept_units
    """
    if not isinstance(node, dict):
        return 0

    node_type = node.get('type')

    # Atomic semantic nodes: 1 concept_unit each
    if node_type in ('constraint', 'permission', 'prohibition', 'reference', 'gate'):
        return 1

    # Literals: 0 (structural, not semantic)
    if node_type == 'literal':
        return 0

    # Operations: depends on operator
    if node_type == 'operation':
        op = node.get('op')
        operands = node.get('operands', [])
        operand_units = [count_concept_units(o) for o in operands]

        if op == 'AND':
            return sum(operand_units)
        elif op == 'OR':
            return max(operand_units) if operand_units else 0
        elif op == 'NOT':
            return operand_units[0] if operand_units else 0
        else:
            return 1 + sum(operand_units)

    # Sequences: sum of elements
    if node_type == 'sequence':
        return sum(count_concept_units(e) for e in node.get('elements', []))

    # Records: sum of field values
    if node_type == 'record':
        return sum(count_concept_units(v) for v in node.get('fields', {}).values())

    # Default for expansion dicts (from SPC decoder)
    if 'summary' in node and 'full' in node:
        # Simple rule expansion: 1 constraint + 1 reference = 2
        return 2

    return 0


def measure_semantic_density(
    pointer: str,
    expansion: str,
    ir_node: dict = None,
    tokenizer_id: str = "o200k_base"
) -> Dict:
    """Measure σ, Df, and H(X|S) from real data per Q33.

    The derivation from Q33:
    - σ := N / H(X) (semantic density = concept_units / baseline_tokens)
    - Df := log(N) / log(σ) (fractal dimension)
    - σ^Df = N (tautology by construction - verification check)
    - CDR = N / H(X|S) (concept density ratio)

    Args:
        pointer: SPC pointer (e.g., "C3", "法")
        expansion: Full text expansion
        ir_node: GOV_IR node for concept_unit counting (optional)
        tokenizer_id: Tokenizer encoding name

    Returns:
        Dict with σ, Df, H_X, H_X_given_S, concept_units, CDR
    """
    # Step 1: Measure entropies (token counts)
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding(tokenizer_id)
            H_X = len(enc.encode(expansion))         # H(X) = baseline tokens
            H_X_given_S = len(enc.encode(pointer))   # H(X|S) = pointer tokens
        except Exception:
            # Fallback on any tiktoken error
            H_X = len(expansion) // 4
            H_X_given_S = len(pointer)
    else:
        # Fallback: character-based estimate
        H_X = len(expansion) // 4
        H_X_given_S = len(pointer)

    # Step 2: Count concept_units
    if ir_node:
        N = count_concept_units(ir_node)
    else:
        # Estimate from expansion: ~1 concept per sentence
        N = max(1, expansion.count('.') + expansion.count(';') + 1)

    # Step 3: Compute σ (semantic density)
    sigma = N / H_X if H_X > 0 else 0

    # Step 4: Compute Df (fractal dimension)
    if sigma > 0 and sigma != 1 and N > 0:
        Df = math.log(N) / math.log(sigma)
    else:
        Df = 1.0  # Degenerate case

    # Step 5: Verify σ^Df ≈ N
    sigma_Df = sigma ** Df if sigma > 0 else 0

    # Step 6: Compute CDR
    CDR = N / H_X_given_S if H_X_given_S > 0 else float('inf')

    # Step 7: Mutual information
    I_X_S = H_X - H_X_given_S

    return {
        "pointer": pointer,
        "H_X": H_X,                    # Baseline entropy (tokens)
        "H_X_given_S": H_X_given_S,    # Conditional entropy (tokens)
        "I_X_S": I_X_S,                # Mutual information (tokens saved)
        "N": N,                        # concept_units
        "sigma": round(sigma, 4),      # Semantic density
        "Df": round(Df, 4),            # Fractal dimension
        "sigma_Df": round(sigma_Df, 4),# Should equal N
        "CDR": round(CDR, 2),          # Concept Density Ratio
        "compression_ratio": round(H_X / H_X_given_S, 2) if H_X_given_S > 0 else float('inf'),
        "verification": abs(sigma_Df - N) < 0.01  # σ^Df = N check
    }


@dataclass
class DensityMetrics:
    """Tracks semantic density metrics per symbol."""

    # CDR tracking
    cdr_samples: List[float] = field(default_factory=list)

    # ECR tracking
    total_expansions: int = 0
    correct_expansions: int = 0

    # Full measurement samples
    density_measurements: List[Dict] = field(default_factory=list)

    def record_expansion(
        self,
        pointer: str,
        expansion: str,
        ir_node: dict = None,
        correct: bool = True
    ):
        """Record a single expansion with full Q33 measurement."""
        # Compute full density metrics
        measurement = measure_semantic_density(pointer, expansion, ir_node)
        self.density_measurements.append(measurement)

        # CDR tracking
        if measurement["H_X_given_S"] > 0:
            self.cdr_samples.append(measurement["CDR"])

        # ECR tracking
        self.total_expansions += 1
        if correct:
            self.correct_expansions += 1

    @property
    def cdr(self) -> float:
        """Average Concept Density Ratio."""
        if not self.cdr_samples:
            return 0.0
        return sum(self.cdr_samples) / len(self.cdr_samples)

    @property
    def ecr(self) -> float:
        """Exact Correctness Rate."""
        if self.total_expansions == 0:
            return 1.0  # No expansions = no errors
        return self.correct_expansions / self.total_expansions

    @property
    def avg_compression(self) -> float:
        """Average compression ratio."""
        ratios = [m["compression_ratio"] for m in self.density_measurements
                  if m["compression_ratio"] != float('inf')]
        return sum(ratios) / len(ratios) if ratios else 0.0


class SPCMetricsTracker:
    """Global tracker for SPC metrics.

    Tracks:
    - Per-symbol CDR, ECR, compression
    - Global aggregates
    - Blanket alignment status (per Q35)
    """

    def __init__(self):
        self.per_symbol: Dict[str, DensityMetrics] = {}
        self.global_metrics = DensityMetrics()
        self.blanket_status: str = "UNSYNCED"  # Per Q35

    def record(
        self,
        pointer: str,
        expansion: str,
        ir_node: dict = None,
        correct: bool = True
    ) -> Dict:
        """Record expansion metrics with Q33 measurement.

        Per Q35: CDR is only defined when blanket_status == ALIGNED.
        Recording without alignment returns an error.

        Args:
            pointer: SPC pointer
            expansion: Full expansion text
            ir_node: Optional GOV_IR node for concept_unit counting
            correct: Whether expansion was correct

        Returns:
            Dict with status or error
        """
        if self.blanket_status != "ALIGNED":
            # CDR undefined without alignment (Q33/Q35)
            return {"error": "E_BLANKET_NOT_ALIGNED", "blanket_status": self.blanket_status}

        # Global
        self.global_metrics.record_expansion(pointer, expansion, ir_node, correct)

        # Per-symbol (use base pointer)
        base = self._extract_base(pointer)
        if base not in self.per_symbol:
            self.per_symbol[base] = DensityMetrics()
        self.per_symbol[base].record_expansion(pointer, expansion, ir_node, correct)

        return {"status": "recorded", "pointer": pointer}

    def set_blanket_status(self, status: str):
        """Update Markov blanket status.

        Per Q35:
        - ALIGNED: R > τ, stable blanket, semantic transfer permitted
        - DISSOLVED: R < τ, blanket broken, resync required
        - PENDING: R ≈ τ, boundary forming
        - UNSYNCED: No sync attempted
        """
        self.blanket_status = status

    def _extract_base(self, pointer: str) -> str:
        """Extract base pointer (radical or hash prefix)."""
        if pointer.startswith("sha256:"):
            return "HASH"
        return pointer[0] if pointer else "UNKNOWN"

    def get_report(self) -> Dict:
        """Get metrics report with Q33 measurements.

        Returns comprehensive report including:
        - Blanket status (Q35)
        - Global CDR, ECR, compression
        - Per-symbol breakdown
        """
        return {
            "blanket_status": self.blanket_status,
            "global": {
                "cdr": round(self.global_metrics.cdr, 2),
                "ecr": round(self.global_metrics.ecr, 4),
                "avg_compression": round(self.global_metrics.avg_compression, 2),
                "total_expansions": self.global_metrics.total_expansions
            },
            "per_symbol": {
                sym: {
                    "cdr": round(m.cdr, 2),
                    "ecr": round(m.ecr, 4),
                    "avg_compression": round(m.avg_compression, 2),
                    "count": m.total_expansions
                }
                for sym, m in self.per_symbol.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global instance
_tracker: Optional[SPCMetricsTracker] = None


def get_metrics_tracker() -> SPCMetricsTracker:
    """Get or create global metrics tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SPCMetricsTracker()
    return _tracker


# Convenience exports
__all__ = [
    "count_concept_units",
    "measure_semantic_density",
    "DensityMetrics",
    "SPCMetricsTracker",
    "get_metrics_tracker"
]
