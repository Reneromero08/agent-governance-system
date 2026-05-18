"""Phase 4b: Phase Drift Diagnostics

When the hard gate fires, classify the drift type based on which nodes failed.
Each drift type maps to a specific correction context for regeneration.

Drift Types:
    FACTUAL_DECOHERENCE    -> Node 2 (Factual) fail, Nodes 1+3 pass
    COMMONSENSE_VIOLATION  -> Node 1 (COMMONSENSE) fail, Nodes 2+3 pass
    SELF_CONSISTENCY_FAIL  -> Node 3 (SelfConsistency) fail, Nodes 1+2 pass
    FULL_DECOHERENCE       -> All three fail
    PARTIAL_MIXED          -> Two fail, one passes (ambiguous pattern)
    SINGLE_MARGINAL        -> Only one passes, others abstain/marginal

Correction Contexts:
    Each drift type has a specific correction message injected into the
    context window to guide regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from phase4b_fragments import FragmentResult


# ============================================================================
# Drift Type Enum
# ============================================================================

class DriftType(Enum):
    FACTUAL_DECOHERENCE = "factual_decoherence"
    COMMONSENSE_VIOLATION = "commonsense_violation"
    SELF_CONSISTENCY_FAIL = "self_consistency_failure"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    FULL_DECOHERENCE = "full_decoherence"
    PARTIAL_MIXED = "partial_mixed"
    NO_DRIFT = "no_drift"


# ============================================================================
# Correction Context Templates
# ============================================================================

CORRECTION_TEMPLATES: Dict[DriftType, str] = {
    DriftType.FACTUAL_DECOHERENCE: (
        "VERIFICATION FAILED: Your factual claims were inconsistent with verified sources. "
        "Re-check your facts and try again. Specifically: {evidence}"
    ),
    DriftType.COMMONSENSE_VIOLATION: (
        "VERIFICATION FAILED: Your output violated core invariants or commonsense rules. "
        "Review the principles and try again. Specifically: {evidence}"
    ),
    DriftType.SELF_CONSISTENCY_FAIL: (
        "VERIFICATION FAILED: Your output was inconsistent with itself when re-generated "
        "under similar conditions. Generate a more coherent and stable response. "
        "Specifically: {evidence}"
    ),
    DriftType.LOGICAL_CONTRADICTION: (
        "VERIFICATION FAILED: Your output contains logical contradictions "
        "or structural inconsistencies. Re-examine your reasoning. "
        "Specifically: {evidence}"
    ),
    DriftType.FULL_DECOHERENCE: (
        "VERIFICATION FAILED: Your output failed ALL verification channels. "
        "Re-examine the prompt completely and try again with a fundamentally "
        "different approach. Specifically: {evidence}"
    ),
    DriftType.PARTIAL_MIXED: (
        "VERIFICATION FAILED: Your output failed multiple verification checks. "
        "Review and correct the issues before responding. "
        "Specifically: {evidence}"
    ),
    DriftType.NO_DRIFT: (
        "Verification passed. Continue."
    ),
}


# ============================================================================
# Diagnostic Result
# ============================================================================

@dataclass
class DriftDiagnostic:
    """Result of phase drift classification."""
    drift_type: DriftType
    failed_fragments: List[str]     # Names of fragments that failed
    passed_fragments: List[str]     # Names of fragments that passed
    abstained_fragments: List[str]  # Names of fragments that abstained
    evidence_summary: str           # Aggregated evidence for correction
    correction_context: str         # Full correction context message
    consensus_ratio: float
    grad_S: float
    fragment_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "drift_type": self.drift_type.value,
            "failed_fragments": self.failed_fragments,
            "passed_fragments": self.passed_fragments,
            "abstained_fragments": self.abstained_fragments,
            "evidence_summary": self.evidence_summary[:300],
            "correction_context": self.correction_context[:300],
            "consensus_ratio": round(self.consensus_ratio, 4),
            "grad_S": round(self.grad_S, 4),
        }


# ============================================================================
# Classifier
# ============================================================================

def classify_drift(
    fragment_results: List[FragmentResult],
    consensus_ratio: float,
    grad_S: float,
) -> DriftDiagnostic:
    """Classify the phase drift type from fragment results.

    Examines which fragments failed/passed and maps to a drift type.
    Generates the appropriate correction context for regeneration.

    Args:
        fragment_results: List of FragmentResult from all active fragments
        consensus_ratio: Current consensus ratio (passing / active)
        grad_S: Current dissonance density

    Returns:
        DriftDiagnostic with classification and correction context
    """
    failed = [f for f in fragment_results if f.verdict in ("hard_fail", "soft_fail")]
    passed = [f for f in fragment_results if f.verdict == "pass"]
    abstained = [f for f in fragment_results if f.verdict == "abstain"]

    failed_names = [f.fragment_name for f in failed]
    passed_names = [f.fragment_name for f in passed]
    abstained_names = [f.fragment_name for f in abstained]

    n_active = len(failed) + len(passed)
    n_failed = len(failed)

    # --- Classify drift type ---
    if n_failed == 0:
        drift_type = DriftType.NO_DRIFT
    elif n_failed == 3:
        drift_type = DriftType.FULL_DECOHERENCE
    elif n_failed == 2:
        # Two failures: identify which ones
        has_commonsense = any("COMMONSENSE" in n for n in failed_names)
        has_factual = any("Factual" in n for n in failed_names)
        has_selfconsist = any("SelfConsistency" in n for n in failed_names)
        has_logical = any("Logical" in n for n in failed_names)

        if has_commonsense and has_factual:
            drift_type = DriftType.PARTIAL_MIXED
        elif has_commonsense and has_selfconsist:
            drift_type = DriftType.PARTIAL_MIXED
        elif has_factual and has_selfconsist:
            drift_type = DriftType.FACTUAL_DECOHERENCE  # Prioritize factual
        elif has_logical:
            drift_type = DriftType.LOGICAL_CONTRADICTION
        else:
            drift_type = DriftType.PARTIAL_MIXED
    elif n_failed == 1:
        # Single failure: identify the specific fragment
        fname = failed[0].fragment_name
        if "COMMONSENSE" in fname:
            drift_type = DriftType.COMMONSENSE_VIOLATION
        elif "Factual" in fname:
            drift_type = DriftType.FACTUAL_DECOHERENCE
        elif "SelfConsistency" in fname:
            drift_type = DriftType.SELF_CONSISTENCY_FAIL
        elif "Logical" in fname:
            drift_type = DriftType.LOGICAL_CONTRADICTION
        else:
            drift_type = DriftType.COMMONSENSE_VIOLATION
    else:
        drift_type = DriftType.NO_DRIFT

    # --- Build evidence summary ---
    evidence_parts = []
    for f in failed:
        evidence_parts.append(f"{f.fragment_name}: {f.evidence[:100]}")
    evidence_summary = " | ".join(evidence_parts)

    # --- Build correction context ---
    template = CORRECTION_TEMPLATES.get(drift_type, CORRECTION_TEMPLATES[DriftType.FULL_DECOHERENCE])
    correction_context = template.format(evidence=evidence_summary)

    return DriftDiagnostic(
        drift_type=drift_type,
        failed_fragments=failed_names,
        passed_fragments=passed_names,
        abstained_fragments=abstained_names,
        evidence_summary=evidence_summary,
        correction_context=correction_context,
        consensus_ratio=consensus_ratio,
        grad_S=grad_S,
        fragment_results={
            f.fragment_name: {"verdict": f.verdict, "score": f.score}
            for f in fragment_results
        },
    )


def build_correction_messages(
    diagnostic: DriftDiagnostic,
    original_prompt: str,
    failed_output: str,
) -> List[dict]:
    """Build chat messages for context correction during hard gate.

    Returns list of message dicts to inject before regeneration.
    """
    return [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": failed_output},
        {"role": "system", "content": diagnostic.correction_context},
    ]


# ============================================================================
# Diagnostic Accuracy Tracker
# ============================================================================

class DiagnosticTracker:
    """Track drift classification accuracy for reporting."""

    def __init__(self):
        self.classifications: List[dict] = []
        self.manual_reviews: List[dict] = []

    def record_classification(
        self,
        prompt_id: str,
        diagnostic: DriftDiagnostic,
        was_correct: bool,
    ):
        self.classifications.append({
            "prompt_id": prompt_id,
            "drift_type": diagnostic.drift_type.value,
            "failed_fragments": diagnostic.failed_fragments,
            "output_was_wrong": was_correct,
            "hard_gate_correctly_fired": was_correct,  # Hard gate should fire on wrong outputs
        })

    def record_manual_review(
        self,
        prompt_id: str,
        classified_drift: str,
        actual_issue: str,
        matches: bool,
    ):
        self.manual_reviews.append({
            "prompt_id": prompt_id,
            "classified_drift": classified_drift,
            "actual_issue": actual_issue,
            "matches": matches,
        })

    def get_stats(self) -> dict:
        if not self.classifications:
            return {"n_classifications": 0}

        hard_gates = self.classifications
        n_hard = len(hard_gates)
        n_correctly_fired = sum(1 for c in hard_gates if c["hard_gate_correctly_fired"])

        reviews = self.manual_reviews
        n_reviews = len(reviews)
        n_matched = sum(1 for r in reviews if r["matches"])

        return {
            "n_classifications": n_hard,
            "hard_gate_precision": round(n_correctly_fired / max(n_hard, 1), 4),
            "n_manual_reviews": n_reviews,
            "classification_accuracy": round(n_matched / max(n_reviews, 1), 4) if n_reviews else None,
        }
