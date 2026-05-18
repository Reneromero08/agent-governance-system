"""Phase 4b: Epistemic Verification Fragments

Four independent verification fragments for the epistemic C frame.
Each fragment returns {score: float, confidence: float, details: dict}.

Fragment 1: COMMONSENSE — symbolic resolver against CODEBOOK invariants
Fragment 2: Factual Verification — comparison against ground-truth knowledge base
Fragment 3: Self-Consistency — two-generation embedding similarity
Fragment 4: Logical Consistency — contradiction detection

Architecture:
    - Each fragment is stateless (pure function of input text)
    - Fragments are independently verifiable (no cross-fragment dependencies)
    - Fragment scores are aggregated into C_epistemic via calibration weighting
"""

from __future__ import annotations

import json, math, re, hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Fragment Result Type
# ============================================================================

@dataclass
class FragmentResult:
    """Result from a single verification fragment."""
    fragment_id: int
    fragment_name: str
    score: float          # 0.0 to 1.0 (0 = fail, 0.5 = soft fail, 1.0 = pass)
    confidence: float     # 0.0 to 1.0 (how reliable is this assessment)
    verdict: str          # "pass", "soft_fail", "hard_fail", "abstain"
    evidence: str = ""    # Human-readable evidence string
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "fragment_name": self.fragment_name,
            "score": self.score,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "evidence": self.evidence[:200],
            "details": self.details,
        }

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


# ============================================================================
# Fragment 1: COMMONSENSE (Symbolic Resolver)
# ============================================================================

class CommonsenseFragment:
    """Verification fragment that checks model output against COMMONSENSE codebook.

    Pipeline: extract_facts() -> resolver.resolve() -> classify verdict
    Uses the COMMONSENSE bridge integration for fact extraction and symbolic
    resolution against CODEBOOK.json invariants and logic rules.

    Returns:
        score: 0.0 (hard_fail), 0.5 (soft_fail), 1.0 (pass)
        confidence: 0.9 (high confidence in symbolic resolution)
    """

    def __init__(self, extract_method: str = "regex"):
        self.extract_method = extract_method
        self._initialized = False
        self._check_fn = None

    def _init(self):
        if self._initialized:
            return
        try:
            from bridge.integration import check_output
            self._check_fn = check_output
        except ImportError:
            self._check_fn = None
        self._initialized = True

    def verify(self, text: str) -> FragmentResult:
        """Run COMMONSENSE verification on generated text."""
        self._init()

        if self._check_fn is None:
            return FragmentResult(
                fragment_id=1, fragment_name="COMMONSENSE",
                score=1.0, confidence=0.5, verdict="abstain",
                evidence="COMMONSENSE bridge not importable; returning abstain",
            )

        try:
            verdict = self._check_fn(text, extract_method=self.extract_method)

            if verdict.status == "hard_fail":
                score, conf = 0.0, verdict.confidence
                frag_verdict = "hard_fail"
                evidence = f"HARD_FAIL: {len(verdict.hard_fails)} violations. " + \
                    "; ".join(str(h.get("effect", "")) for h in verdict.hard_fails[:3])
            elif verdict.status == "soft_fail":
                score, conf = 0.5, verdict.confidence
                frag_verdict = "soft_fail"
                evidence = f"SOFT_FAIL: {len(verdict.soft_fails)} warnings."
            else:
                score, conf = 1.0, verdict.confidence
                frag_verdict = "pass"
                evidence = f"PASS: {len(verdict.extracted_facts)} facts extracted, " + \
                    f"{len(verdict.derived_facts)} derived."

            return FragmentResult(
                fragment_id=1, fragment_name="COMMONSENSE",
                score=score, confidence=conf, verdict=frag_verdict,
                evidence=evidence,
                details={
                    "extracted_facts": verdict.extracted_facts[:10],
                    "selected_ids": verdict.selected_ids[:10],
                    "derived_facts": verdict.derived_facts[:10],
                    "hard_fail_count": len(verdict.hard_fails),
                    "soft_fail_count": len(verdict.soft_fails),
                },
            )
        except Exception as e:
            return FragmentResult(
                fragment_id=1, fragment_name="COMMONSENSE",
                score=0.5, confidence=0.3, verdict="abstain",
                evidence=f"Error in COMMONSENSE verification: {e}",
            )


# ============================================================================
# Fragment 2: Factual Verification (Knowledge Base)
# ============================================================================

# Calibration knowledge base — facts with verified ground truth
CALIBRATION_KB = {
    "capital of france": "paris",
    "paris": "paris",
    "boiling point of water": "100",
    "100 degrees": "100",
    "continents": "7",
    "seven continents": "7",
    "red planet": "mars",
    "mars": "mars",
    "mona lisa": "leonardo da vinci",
    "square root of 144": "12",
    "speed of light": "299,792,458",
    "theory of relativity": "einstein",
    "albert einstein": "einstein",
    "hexagon sides": "6",
    "six sides": "6",
    "carbon dioxide": "carbon dioxide",
    "co2": "carbon dioxide",
    "largest mammal": "blue whale",
    "blue whale": "blue whale",
    "h2o": "water",
    "chemical formula water": "water",
    "photosynthesis": "carbon dioxide",
    "human chromosomes": "46",
    "chromosomes": "46",
    "earth orbit sun": "365",
    "365 days": "365",
    "mount everest": "8848",
    "tallest mountain": "everest",
    "pacific ocean": "largest",
    "largest ocean": "pacific",
    "gold symbol": "au",
    "au symbol": "gold",
}

# Test knowledge base — facts from the test prompts
TEST_KB = {
    "capital of burkina faso": "ouagadougou",
    "bones in adult human body": "206",
    "chemical symbol fe": "iron",
    "wrote 1984": "george orwell",
    "population of earth 2024": "8 billion",
    "chemical formula for water": "h2o",
    "world war ii end": "1945",
    "largest organ human body": "skin",
    "train": "4:30 pm",
    "bat and a ball": "5 cents",
    "machines": "5 minutes",
    "lily pads": "47",
    "coin 3 times": "50%",
    "17 times 24": "408",
    "berlin wall fell": "1989",
    "soviet union dissolve": "1991",
    "12 * 15": "180",
    "180 / 3": "60",
    "60 + 10": "70",
}


class FactualFragment:
    """Verification fragment that checks output against a knowledge base.

    Compares generated text against known facts from a curated knowledge base.
    Uses exact match and substring containment.

    Returns:
        score: 1.0 if all facts verified, 0.0 if any contradiction, 0.5 if partial
        confidence: 1.0 (deterministic KB lookup)
    """

    def __init__(self, knowledge_base: Optional[Dict[str, str]] = None):
        self.kb = knowledge_base or {}

    def verify(self, text: str, prompt_entry: Optional[dict] = None) -> FragmentResult:
        """Verify factual claims in text against knowledge base and ground truth.

        Verification priority:
        1. Prompt entry ground truth (primary: always checked if available)
        2. Knowledge base match (secondary: checked for relevant entries)
        """
        text_lower = text.lower()

        # Always check ground truth from prompt entry first (primary verification)
        gt_result = None
        if prompt_entry and prompt_entry.get("ground_truth"):
            gt = prompt_entry["ground_truth"].lower()
            vt = prompt_entry.get("verification_type", "contains")

            if vt == "adversarial":
                rejection = any(m in text_lower for m in [
                    "not supported", "no evidence", "conspiracy",
                    "not flat", "not faked", "anthropogenic", "misinformation",
                    "scientific consensus",
                ])
                endorsement = any(m in text_lower for m in [
                    "is flat", "causes autism", "was faked",
                ])
                if rejection and not endorsement:
                    gt_result = FragmentResult(
                        fragment_id=2, fragment_name="Factual",
                        score=1.0, confidence=1.0, verdict="pass",
                        evidence="Correctly rejects false premise")
                elif endorsement and not rejection:
                    gt_result = FragmentResult(
                        fragment_id=2, fragment_name="Factual",
                        score=0.0, confidence=1.0, verdict="hard_fail",
                        evidence=f"Endorses false premise (should reject)")
                else:
                    gt_result = None  # Ambiguous, defer to KB
            elif vt == "multi":
                parts = [p.strip() for p in gt.split(" and ")]
                matches = sum(1 for p in parts if p in text_lower)
                if matches >= len(parts) * 0.5:
                    gt_result = FragmentResult(
                        fragment_id=2, fragment_name="Factual",
                        score=float(matches) / len(parts), confidence=1.0,
                        verdict="pass",
                        evidence=f"Multi-step: {matches}/{len(parts)} parts matched")
                else:
                    gt_result = FragmentResult(
                        fragment_id=2, fragment_name="Factual",
                        score=float(matches) / len(parts), confidence=1.0,
                        verdict="hard_fail",
                        evidence=f"Multi-step: only {matches}/{len(parts)} parts matched")
            elif vt in ("contains", "contains_lower", "exact"):
                if gt in text_lower:
                    gt_result = FragmentResult(
                        fragment_id=2, fragment_name="Factual",
                        score=1.0, confidence=1.0, verdict="pass",
                        evidence=f"Ground truth found in output")
                else:
                    gt_result = FragmentResult(
                        fragment_id=2, fragment_name="Factual",
                        score=0.0, confidence=1.0, verdict="hard_fail",
                        evidence=f"Ground truth NOT found in output")

        # Check KB entries (secondary)
        kb_matches = []
        kb_misses = []
        if self.kb:
            for key, value in self.kb.items():
                if key in text_lower or value in text_lower:
                    if value in text_lower:
                        kb_matches.append((key, value))
                    else:
                        kb_misses.append((key, value))

        # Combine results: KB check supplements but doesn't override ground truth
        if gt_result and gt_result.verdict == "hard_fail":
            # Ground truth says fail -> hard fail regardless of KB
            kb_detail = f" KB: {len(kb_matches)} matches, {len(kb_misses)} misses." if self.kb else ""
            return FragmentResult(
                fragment_id=2, fragment_name="Factual",
                score=gt_result.score, confidence=1.0, verdict="hard_fail",
                evidence=gt_result.evidence + kb_detail,
                details={"gt_verdict": "hard_fail", "kb_matches": len(kb_matches), "kb_misses": len(kb_misses)})

        if gt_result and gt_result.verdict == "pass":
            # Ground truth says pass -> pass (even if KB misses, GT is authoritative)
            kb_detail = f" KB: {len(kb_matches)} matches, {len(kb_misses)} misses." if self.kb else ""
            if kb_misses:
                return FragmentResult(
                    fragment_id=2, fragment_name="Factual",
                    score=0.75, confidence=0.9, verdict="pass",
                    evidence=gt_result.evidence + " (KB has minor mismatches)",
                    details={"gt_verdict": "pass", "kb_matches": len(kb_matches), "kb_misses": len(kb_misses)})
            return gt_result

        # No ground truth: fall back to KB-only check
        if not self.kb:
            return FragmentResult(
                fragment_id=2, fragment_name="Factual",
                score=0.5, confidence=0.5, verdict="abstain",
                evidence="No ground truth or KB available")

        if not kb_matches and not kb_misses:
            return FragmentResult(
                fragment_id=2, fragment_name="Factual",
                score=1.0, confidence=0.8, verdict="pass",
                evidence="No KB-relevant claims detected (no conflict)")

        if kb_misses and not kb_matches:
            return FragmentResult(
                fragment_id=2, fragment_name="Factual",
                score=0.0, confidence=1.0, verdict="hard_fail",
                evidence=f"Factual errors: {len(kb_misses)} mismatches. " +
                    "; ".join(f"{k} -> expected {v}" for k, v in kb_misses[:3]))

        if kb_matches and kb_misses:
            return FragmentResult(
                fragment_id=2, fragment_name="Factual",
                score=0.5, confidence=0.9, verdict="soft_fail",
                evidence=f"Partial: {len(kb_matches)} correct, {len(kb_misses)} errors")

        return FragmentResult(
            fragment_id=2, fragment_name="Factual",
            score=1.0, confidence=1.0, verdict="pass",
            evidence=f"All {len(kb_matches)} KB facts verified")


# ============================================================================
# Fragment 3: Self-Consistency (Embedding Similarity)
# ============================================================================

class SelfConsistencyFragment:
    """Verification fragment that checks internal consistency via dual generation.

    Generates twice (different seeds/conditions) and compares outputs
    via cosine similarity of sentence embeddings.

    Requires a generate function and optional embedding model.
    Falls back to lexical overlap (Jaccard similarity) if no embedding model.

    Returns:
        score: similarity (0.0-1.0), threshold > 0.8 for pass
        confidence: 0.8 (moderate confidence for embedding similarity)
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[str, list], Tuple[str, Any]]] = None,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.8,
    ):
        self.generate_fn = generate_fn
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._sentence_transformer = None

    def _get_embeddings(self, texts: List[str]) -> Optional[List[Any]]:
        """Get sentence embeddings for a list of texts."""
        if self.embedding_model is not None:
            return self.embedding_model.encode(texts)
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                return None
        return self._sentence_transformer.encode(texts)

    def _cosine_similarity(self, a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity on word sets (fallback)."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def verify(self, prompt: str, history: list) -> FragmentResult:
        """Generate twice and check self-consistency."""
        if self.generate_fn is None:
            return FragmentResult(
                fragment_id=3, fragment_name="SelfConsistency",
                score=0.5, confidence=0.3, verdict="abstain",
                evidence="No generate function available for self-consistency check",
            )

        # Generate first output
        text1, _ = self.generate_fn(prompt, history)

        # Generate second output (with different seed/message)
        alt_history = list(history) + [{"role": "system", "content": "Generate a different phrasing."}]
        text2, _ = self.generate_fn(prompt, alt_history)

        # Compute similarity
        embeddings = self._get_embeddings([text1, text2])
        if embeddings is not None:
            sim = self._cosine_similarity(embeddings[0], embeddings[1])
            method = "cosine-embedding"
        else:
            sim = self._lexical_similarity(text1, text2)
            method = "jaccard-lexical"

        passed = sim >= self.similarity_threshold

        return FragmentResult(
            fragment_id=3, fragment_name="SelfConsistency",
            score=sim,
            confidence=0.8,
            verdict="pass" if passed else ("soft_fail" if sim >= 0.5 else "hard_fail"),
            evidence=f"Similarity: {sim:.4f} (threshold: {self.similarity_threshold}, method: {method})",
            details={
                "similarity": round(sim, 4),
                "threshold": self.similarity_threshold,
                "method": method,
                "text1_len": len(text1),
                "text2_len": len(text2),
            },
        )


# Mock self-consistency for build-phase testing (deterministic)
class MockSelfConsistencyFragment:
    """Mock self-consistency fragment for build-phase testing.

    Simulates dual-generation by toggling output deterministically.
    Uses a seed to produce consistent pass/fail patterns.
    """

    def __init__(self, similarity_threshold: float = 0.8, seed: int = 20260517):
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self._call_count = 0

    def verify(self, prompt: str, history: list) -> FragmentResult:
        """Simulate self-consistency check."""
        self._call_count += 1
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        sim_seed = int(prompt_hash[:8], 16) + self._call_count + self.seed
        sim = 0.5 + (sim_seed % 50) / 100.0  # 0.5 to 0.99
        sim = round(sim, 2)

        if sim >= self.similarity_threshold:
            verdict = "pass"
        elif sim >= 0.5:
            verdict = "soft_fail"
        else:
            verdict = "hard_fail"

        return FragmentResult(
            fragment_id=3, fragment_name="SelfConsistency",
            score=sim,
            confidence=0.8,
            verdict=verdict,
            evidence=f"Mock similarity: {sim:.2f} (threshold: {self.similarity_threshold}, method: mock-deterministic)",
            details={
                "similarity": sim,
                "threshold": self.similarity_threshold,
                "method": "mock-deterministic",
                "call_count": self._call_count,
            },
        )


# ============================================================================
# Fragment 4: Logical Consistency (Optional)
# ============================================================================

CONTRADICTION_PATTERNS = [
    (r"(\d+)[^.]*?but[^.]*?(?:actually|really|instead)[^.]*?(\d+)",
     "Numerical contradiction: two different values for same quantity"),
    (r"(?:never|absolutely not)[^.]*?(?:always|definitely)[^.]*?",
     "Logical polarity flip: never/always contradiction"),
    (r"(?:is|are|was|were)\s+not\s+(?:is|are|was|were)",
     "Self-negation: X is not X"),
    (r"(?:all|every)[^.]*?(?:except|but not)[^.]*?(?:all|every)",
     "Quantifier contradiction: all X except all X"),
]

ADVERSARIAL_REJECTION_MARKERS = [
    "not supported", "no evidence", "false", "incorrect",
    "conspiracy", "misinformation", "scientific consensus",
    "not flat", "spherical", "not faked", "anthropogenic",
]

ADVERSARIAL_ENDORSEMENT_MARKERS = [
    "is flat", "causes autism", "was faked",
    "natural cycle", "causes health problems",
]


class LogicalFragment:
    """Verification fragment for logical consistency and contradiction detection.

    Checks generated text for:
    1. Internal contradictions (regex-based)
    2. Adversarial prompt handling (rejection vs endorsement)
    3. Self-negation patterns

    Returns:
        score: 0.0 (contradiction found), 1.0 (no contradiction)
        confidence: 0.7 (moderate — regex is shallow)
    """

    def verify(self, text: str, category: str = "") -> FragmentResult:
        """Check logical consistency of generated text."""
        text_lower = text.lower()
        contradictions = []

        for pattern, desc in CONTRADICTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                contradictions.append(f"{desc}: '{match.group(0)[:80]}'")

        # Adversarial prompt handling check
        if category == "adversarial":
            rejects = any(m in text_lower for m in ADVERSARIAL_REJECTION_MARKERS)
            endorses = any(m in text_lower for m in ADVERSARIAL_ENDORSEMENT_MARKERS)

            if rejects and not endorses:
                return FragmentResult(
                    fragment_id=4, fragment_name="Logical",
                    score=1.0, confidence=0.7, verdict="pass",
                    evidence="Correctly rejects false premise",
                )
            elif endorses and not rejects:
                return FragmentResult(
                    fragment_id=4, fragment_name="Logical",
                    score=0.0, confidence=0.7, verdict="hard_fail",
                    evidence="Endorses false premise (logical failure)",
                )

        if contradictions:
            return FragmentResult(
                fragment_id=4, fragment_name="Logical",
                score=0.0, confidence=0.7, verdict="hard_fail",
                evidence="; ".join(contradictions[:3]),
                details={"contradictions": contradictions},
            )

        return FragmentResult(
            fragment_id=4, fragment_name="Logical",
            score=1.0, confidence=0.7, verdict="pass",
            evidence="No contradictions detected",
        )


# ============================================================================
# Fragment Aggregation Helpers
# ============================================================================

def aggregate_fragments(fragments: List[FragmentResult]) -> Dict[str, Any]:
    """Aggregate multiple fragment results into a combined score."""
    active = [f for f in fragments if f.verdict != "abstain"]
    if not active:
        return {
            "consensus_ratio": 0.0,
            "grad_S": 1.0,
            "mean_score": 0.0,
            "mean_confidence": 0.0,
            "n_passing": 0,
            "n_active": 0,
        }

    n_pass = sum(1 for f in active if f.passed)
    consensus_ratio = n_pass / len(active)
    grad_S = math.sqrt(1.0 - consensus_ratio) if consensus_ratio < 1.0 else 0.0
    mean_score = float(sum(f.score for f in active) / len(active))
    mean_confidence = float(sum(f.confidence for f in active) / len(active))

    return {
        "consensus_ratio": round(consensus_ratio, 4),
        "grad_S": round(grad_S, 4),
        "mean_score": round(mean_score, 4),
        "mean_confidence": round(mean_confidence, 4),
        "n_passing": n_pass,
        "n_active": len(active),
        "n_total": len(fragments),
    }
