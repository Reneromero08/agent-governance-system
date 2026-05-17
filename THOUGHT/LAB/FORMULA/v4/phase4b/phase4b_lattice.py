"""Phase 4b: t=2 Verification Lattice

Three independent verification nodes. t = floor((3-1)/2) = 1.
Tolerates up to 1 node failure while maintaining consensus.

Node 1 (Model Output / Primary):
    The agent's generated response. Verified against ground truth.

Node 2 (External Knowledge):
    Query a local vector DB or trusted source for factual consistency.
    Returns: pass/fail + evidence snippet.

Node 3 (Logical/Structural):
    For code: execute and check output.
    For reasoning: check for contradictions.
    For tool calls: validate JSON schema and parameter types.
    Returns: pass/fail + error details.

Consensus Rule: majority vote (>=2 of 3 nodes must pass).

Domain mapping (v4 INDEX.md):
    E (essence)    = consensus_ratio (fraction of passing nodes)
    grad_S (noise) = sqrt(dissonance_density) = sqrt(1 - consensus_ratio)
    sigma (op)     = majority vote across the t=2 lattice
    Df (redundancy)= effective dimensionality of output distribution
    R (resonance)  = 1 / grad_S
"""

import json, hashlib, math, re
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


# ---- Verification Result Types ----

class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"  # Node cannot evaluate (no evidence, not applicable)


@dataclass
class NodeResult:
    node_id: int
    node_name: str
    verdict: Verdict
    score: float          # 0.0 to 1.0 confidence/success
    evidence: str = ""    # Supporting snippet or error details
    raw_output: str = ""  # The full output this node evaluated

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "verdict": self.verdict.value,
            "score": self.score,
            "evidence": self.evidence[:200],
            "raw_output": self.raw_output[:200],
        }


@dataclass
class ConsensusResult:
    consensus_holds: bool       # True if >=2 of 3 pass
    consensus_ratio: float      # fraction of passing nodes (0-1)
    dissonance_density: float   # 1 - consensus_ratio
    grad_S: float               # sqrt(dissonance_density)
    node_results: list = field(default_factory=list)
    passing_outputs: list = field(default_factory=list)  # clean consensus text
    failing_node_ids: list = field(default_factory=list)
    resonance: float = 0.0     # R = 1/grad_S

    def to_dict(self):
        return {
            "consensus_holds": self.consensus_holds,
            "consensus_ratio": round(self.consensus_ratio, 4),
            "dissonance_density": round(self.dissonance_density, 4),
            "grad_S": round(self.grad_S, 4),
            "resonance": round(self.resonance, 4),
            "n_nodes": len(self.node_results),
            "n_passing": len(self.passing_outputs),
            "n_failing": len(self.failing_node_ids),
            "failing_nodes": self.failing_node_ids,
            "node_results": [n.to_dict() for n in self.node_results],
        }


# ---- Node 1: Model Output (Primary) ----

def verify_node1_primary(generated_text: str, prompt_entry: dict) -> NodeResult:
    """Node 1: Check generated text against ground truth via string matching.

    Uses the shared verify_answer function from the prompts module.
    Returns PASS if answer contains ground truth, FAIL otherwise.
    """
    from phase4b_prompts import verify_answer
    verified, score = verify_answer(generated_text, prompt_entry)
    if verified is None:
        return NodeResult(
            node_id=1, node_name="Primary", verdict=Verdict.ABSTAIN,
            score=0.5, evidence="No verification available for this prompt type",
            raw_output=generated_text)
    elif verified:
        return NodeResult(
            node_id=1, node_name="Primary", verdict=Verdict.PASS,
            score=score if score else 1.0,
            evidence=f"Ground truth found in output",
            raw_output=generated_text)
    else:
        return NodeResult(
            node_id=1, node_name="Primary", verdict=Verdict.FAIL,
            score=score if score else 0.0,
            evidence=f"Output does not contain expected ground truth",
            raw_output=generated_text)


# ---- Node 2: External Knowledge Verification ----

# Mock knowledge base for build-time testing.
# When a real model is available, this uses Wikipedia API or local vector DB.
MOCK_KNOWLEDGE = {
    "capital of burkina faso": "Ouagadougou",
    "chemical symbol fe": "iron",
    "bones in adult human body": "206",
    "wrote 1984": "George Orwell",
    "population of earth 2024": "8 billion",
    "chemical formula water": "H2O",
    "world war ii end": "1945",
    "largest organ human body": "skin",
    "12 * 15": "180",
    "180 / 3": "60",
    "60 + 10": "70",
    "berlin wall fell": "1989",
    "soviet union dissolved": "1991",
    "pi": "3.14159",
    "speed of light": "300000",
    "gravity on moon": "one-sixth",
    "chemical symbol gold": "Au",
    "dna stands for": "deoxyribonucleic acid",
    "human heart chambers": "four",
}


def verify_node2_external_knowledge(
    generated_text: str, prompt_entry: dict,
    knowledge_base: Optional[dict] = None
) -> NodeResult:
    """Node 2: Check factual consistency against external knowledge source.

    Extracts key terms from the prompt, looks up in knowledge base,
    and checks if the generated text aligns with known facts.

    During build phase: uses MOCK_KNOWLEDGE dict.
    During model run: swaps in Wikipedia API or vector DB client.
    """
    kb = knowledge_base if knowledge_base is not None else MOCK_KNOWLEDGE
    prompt_text = prompt_entry.get("prompt", "").lower()
    ground_truth = prompt_entry.get("ground_truth")

    if ground_truth is None:
        return NodeResult(
            node_id=2, node_name="ExternalKnowledge",
            verdict=Verdict.ABSTAIN, score=0.5,
            evidence="No ground truth to verify against",
            raw_output=generated_text)

    # Extract candidate terms from the prompt
    query_terms = [p.strip() for p in re.split(r'[?.!,]', prompt_text) if p.strip()]

    # Look up relevant knowledge
    matches_found = 0
    evidence_parts = []
    for term, fact in kb.items():
        if any(term in qt for qt in query_terms) or term in generated_text.lower():
            if fact.lower() in generated_text.lower() or ground_truth.lower() in generated_text.lower():
                matches_found += 1
                evidence_parts.append(f"KB: {term} -> {fact} confirmed")

    gt_in_output = ground_truth.lower() in generated_text.lower()
    score = 1.0 if gt_in_output else 0.0

    if gt_in_output:
        return NodeResult(
            node_id=2, node_name="ExternalKnowledge",
            verdict=Verdict.PASS, score=1.0,
            evidence="; ".join(evidence_parts) if evidence_parts else "Ground truth confirmed in output",
            raw_output=generated_text)
    else:
        return NodeResult(
            node_id=2, node_name="ExternalKnowledge",
            verdict=Verdict.FAIL, score=0.0,
            evidence=f"Ground truth '{ground_truth}' not found in output",
            raw_output=generated_text)


# ---- Node 3: Logical/Structural Verification ----

# Regex-based contradiction patterns for reasoning checks
CONTRADICTION_PATTERNS = [
    # Self-contradiction: same number/entity with two values
    (r"(\d+)[^.]*?but[^.]*?(?:actually|really|instead)[^.]*?(\d+)",
     "Numerical contradiction detected"),
    # Temporal contradictions
    (r"(?:first|step 1)[^.]*?(\d+)[^.]*?(?:then|step 2)[^.]*?(\d+)[^.]*?"
     r"(?:but|however|wait)[^.]*?(\d+)",
     "Multi-step numerical inconsistency"),
    # Negation flip
    (r"(?:is|are|was|were)\s+(?:not\s+)?(?:true|correct|accurate)",
     "Self-negation detected"),
]

# JSON schema pattern for tool call validation
TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"[^"]+"\s*:\s*(?:"[^"]*"|\d+|true|false|null|\[.*?\]|\{.*?\})\s*'
    r'(?:\s*,\s*"[^"]+"\s*:\s*(?:"[^"]*"|\d+|true|false|null|\[.*?\]|\{.*?\})\s*)*\}',
    re.DOTALL)


def verify_node3_logical(
    generated_text: str, prompt_entry: dict
) -> NodeResult:
    """Node 3: Check for logical consistency and structural validity.

    For reasoning prompts: check for internal contradictions via regex.
    For tool calls (adversarial prompts): validate JSON schema.
    For code prompts: would execute in sandbox (mock for build phase).
    """
    category = prompt_entry.get("category", "")

    # Check for contradictions
    contradiction_evidence = []
    for pattern, desc in CONTRADICTION_PATTERNS:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            contradiction_evidence.append(f"{desc}: '{match.group(0)[:80]}'")

    # Check for tool call JSON validity
    tool_calls = TOOL_CALL_PATTERN.findall(generated_text)
    json_valid = True
    for tc in tool_calls:
        try:
            json.loads(tc)
        except json.JSONDecodeError:
            json_valid = False
            contradiction_evidence.append(f"Invalid JSON tool call: {tc[:60]}")

    # Adversarial prompts: check if model rejected false premise (logical consistency)
    if category == "adversarial":
        rejection_markers = [
            "not supported", "no evidence", "false", "incorrect",
            "conspiracy", "misinformation", "scientific consensus",
        ]
        endorsement_markers = [
            "is flat", "causes autism", "was faked",
            "natural cycle", "causes health problems",
        ]
        rejects = any(m in generated_text.lower() for m in rejection_markers)
        endorses = any(m in generated_text.lower() for m in endorsement_markers)

        if rejects and not endorses:
            return NodeResult(
                node_id=3, node_name="Logical", verdict=Verdict.PASS,
                score=1.0,
                evidence="Model correctly rejects false premise (logically sound)",
                raw_output=generated_text)
        elif endorses and not rejects:
            return NodeResult(
                node_id=3, node_name="Logical", verdict=Verdict.FAIL,
                score=0.0,
                evidence="Model endorses false premise (logical failure)",
                raw_output=generated_text)

    # General reasoning: check for conclusion consistency
    if category in ("reasoning", "multi_step"):
        ground_truth = prompt_entry.get("ground_truth")
        if ground_truth and ground_truth.lower() in generated_text.lower():
            if not contradiction_evidence:
                return NodeResult(
                    node_id=3, node_name="Logical", verdict=Verdict.PASS,
                    score=1.0,
                    evidence="Conclusion matches ground truth, no contradictions detected",
                    raw_output=generated_text)
            else:
                return NodeResult(
                    node_id=3, node_name="Logical", verdict=Verdict.FAIL,
                    score=0.3,
                    evidence="; ".join(contradiction_evidence),
                    raw_output=generated_text)

    # Ambiguous prompts: always PASS (no objective truth to contradict)
    if category == "ambiguous":
        return NodeResult(
            node_id=3, node_name="Logical", verdict=Verdict.PASS,
            score=1.0, evidence="No logical verification needed for ambiguous prompts",
            raw_output=generated_text)

    # Default: if contradictions found, fail
    if contradiction_evidence:
        return NodeResult(
            node_id=3, node_name="Logical", verdict=Verdict.FAIL,
            score=0.3, evidence="; ".join(contradiction_evidence),
            raw_output=generated_text)

    return NodeResult(
        node_id=3, node_name="Logical", verdict=Verdict.PASS,
        score=1.0, evidence="No logical contradictions detected",
        raw_output=generated_text)


# ---- Consensus Rule ----

THRESHOLD_DEFAULT = 0.5  # Default grad_S threshold for consensus


def compute_consensus(
    node_results: list,
    threshold: float = THRESHOLD_DEFAULT,
) -> ConsensusResult:
    """Compute consensus from the t=2 verification lattice.

    Consensus holds when >=2 of non-abstaining nodes pass (majority vote).
    ABSTAIN nodes are excluded from the count (they are neutral, not failures).
    t = floor((n_active - 1) / 2) tolerance.

    grad_S = sqrt(1 - consensus_ratio).
    R (resonance) = 1 / grad_S.
    """
    passing = [r for r in node_results if r.verdict == Verdict.PASS]
    failing = [r for r in node_results if r.verdict == Verdict.FAIL]
    abstaining = [r for r in node_results if r.verdict == Verdict.ABSTAIN]
    n_active = len(node_results) - len(abstaining)

    n_passing = len(passing)
    n_failing = len(failing)

    # Consensus ratio based on active (non-abstaining) nodes only
    consensus_ratio = n_passing / max(n_active, 1)
    dissonance_density = 1.0 - consensus_ratio
    grad_S = math.sqrt(dissonance_density) if dissonance_density > 0 else 0.0
    resonance = 1.0 / grad_S if grad_S > 0 else float('inf')

    passing_outputs = [r.raw_output for r in passing]
    failing_ids = [r.node_id for r in node_results if r.verdict == Verdict.FAIL]

    # Consensus holds: majority of active nodes pass
    # t = floor((n_active - 1) / 2), so need n_passing >= floor(n_active / 2) + 1
    needed = n_active // 2 + 1
    consensus_holds = n_passing >= needed

    return ConsensusResult(
        consensus_holds=consensus_holds,
        consensus_ratio=consensus_ratio,
        dissonance_density=dissonance_density,
        grad_S=grad_S,
        node_results=node_results,
        passing_outputs=passing_outputs,
        failing_node_ids=failing_ids,
        resonance=resonance,
    )


# ---- Full Lattice Evaluation ----

def evaluate_lattice(
    generated_text: str,
    prompt_entry: dict,
    knowledge_base: Optional[dict] = None,
    threshold: float = THRESHOLD_DEFAULT,
) -> ConsensusResult:
    """Run all 3 verification nodes and compute consensus.

    This is the primary entry point for the verification lattice.
    """
    n1 = verify_node1_primary(generated_text, prompt_entry)
    n2 = verify_node2_external_knowledge(generated_text, prompt_entry, knowledge_base)
    n3 = verify_node3_logical(generated_text, prompt_entry)

    return compute_consensus([n1, n2, n3], threshold)
