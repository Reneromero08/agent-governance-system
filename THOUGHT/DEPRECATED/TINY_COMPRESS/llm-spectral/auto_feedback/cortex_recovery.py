"""Cortex-Backed Recovery Module

Bridges the cassette network into the Phase 3.5/4b feedback loops as a
self-populating fact cache and retrieval mechanism.

Architecture:
    1. On verification failure (hard gate): query resident memory for relevant facts
    2. If cache miss: store the ground truth as a new memory
    3. If cache hit: inject retrieved facts into correction context
    4. Facts accumulate across sessions — the knowledge base grows

This replaces external RAG with the repo's own semantic indexing system.
The cassette network indexes all repository content; the resident memory
cassette stores persistent agent memories with vector search.

Usage:
    from cortex_recovery import CortexRecovery
    cr = CortexRecovery(agent_id="phase4b")
    facts = cr.query("capital of France")
    cr.store("The capital of France is Paris.", metadata={"prompt_id": "F1"})
"""

import hashlib, json, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Absolute path to cassette network
_REPO_ROOT = Path(__file__).resolve().parents[5]
_CORTEX_NETWORK = str(_REPO_ROOT / "NAVIGATION" / "CORTEX" / "network")
_CORTEX_SEMANTIC = str(_REPO_ROOT / "NAVIGATION" / "CORTEX" / "semantic")
if _CORTEX_NETWORK not in sys.path:
    sys.path.insert(0, _CORTEX_NETWORK)
if _CORTEX_SEMANTIC not in sys.path:
    sys.path.insert(0, _CORTEX_SEMANTIC)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class CortexRecovery:
    """Fact cache and retrieval backed by the resident memory cassette.

    Stores correct answers as persistent memories. Retrieves them
    via semantic search on subsequent verification failures.

    Also queries the full cassette network (FTS across all indexed
    repo content) for governance-related context.
    """

    def __init__(self, agent_id: str = "auto-feedback"):
        self.agent_id = agent_id
        self._memory = None
        self._cortex = None
        self._hub = None
        self._initialized = False
        self._stats = {"stores": 0, "hits": 0, "misses": 0, "fts_results": 0}

    def _init(self):
        if self._initialized:
            return
        try:
            from memory_cassette import MemoryCassette
            self._memory = MemoryCassette(agent_id=self.agent_id)
        except Exception as e:
            print(f"[CortexRecovery] Memory cassette unavailable: {e}")
            self._memory = None

        try:
            from query import CortexQuery
            self._cortex = CortexQuery()
        except Exception:
            self._cortex = None

        self._initialized = True

    # ---- Memory (fact cache) ----

    def store(self, text: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """Store a fact as a persistent memory. Returns content hash."""
        self._init()
        if self._memory is None:
            return None
        try:
            h, receipt = self._memory.memory_save(
                text, metadata=metadata, agent_id=self.agent_id)
            self._stats["stores"] += 1
            return h
        except Exception as e:
            print(f"[CortexRecovery] Store failed: {e}")
            return None

    def query(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Semantic search for relevant facts in resident memory."""
        self._init()
        if self._memory is None:
            return []
        try:
            results = self._memory.memory_query(query_text, limit=limit,
                                                 agent_id=self.agent_id)
            if results:
                self._stats["hits"] += 1
            else:
                self._stats["misses"] += 1
            return results
        except Exception as e:
            print(f"[CortexRecovery] Query failed: {e}")
            return []

    def recall(self, memory_hash: str) -> Optional[Dict]:
        """Retrieve a specific memory by hash."""
        self._init()
        if self._memory is None:
            return None
        try:
            return self._memory.memory_recall(memory_hash)
        except Exception:
            return None

    def store_fact(self, prompt: str, correct_answer: str,
                   prompt_id: str = "") -> Optional[str]:
        """Store a prompt-answer pair as a memory for future retrieval."""
        fact_text = f"Q: {prompt}\nA: {correct_answer}"
        metadata = {
            "type": "fact",
            "prompt": prompt[:200],
            "answer": correct_answer[:200],
            "prompt_id": prompt_id,
        }
        return self.store(fact_text, metadata=metadata)

    def retrieve_fact(self, prompt: str, limit: int = 3) -> List[str]:
        """Retrieve relevant stored facts for a prompt. Returns answer strings."""
        results = self.query(prompt, limit=limit)
        facts = []
        for r in results:
            h = r.get("hash", "")
            full = self.recall(h)
            if full and "text" in full:
                text = full["text"]
                # Extract answer from "Q: ...\nA: ..." format
                if "\nA: " in text:
                    facts.append(text.split("\nA: ", 1)[1])
                else:
                    facts.append(text[:300])
        return facts

    # ---- FTS (full-text search across all cassettes) ----

    def search_repo(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Full-text search across all indexed repository content."""
        self._init()
        if self._cortex is None:
            return []
        try:
            results = self._cortex.search(query_text, limit=limit)
            self._stats["fts_results"] += len(results)
            return results
        except Exception:
            return []

    def search_governance(self, query_text: str) -> List[str]:
        """Search for governance-related facts (invariants, rules, ADRs)."""
        results = self.search_repo(query_text, limit=5)
        snippets = []
        for r in results:
            snippet = r.get("snippet", "")[:300]
            path = r.get("path", "")
            if snippet:
                snippets.append(f"[{path}] {snippet}")
        return snippets

    # ---- Correction Context Builder ----

    def build_correction_context(self, prompt: str, ground_truth: str = "",
                                  prompt_id: str = "") -> str:
        """Build a correction context message from cortex-retrieved facts.

        Priority:
        1. Stored facts from resident memory (persistent cache)
        2. Ground truth from prompt entry (authoritative)
        3. FTS results from cassette network (governance context)
        """
        parts = []

        # 1. Check resident memory
        facts = self.retrieve_fact(prompt)
        if facts:
            parts.append("Retrieved facts: " + " | ".join(facts[:3]))

        # 2. Ground truth (always include if available)
        if ground_truth:
            parts.append(f"Correct answer: {ground_truth}")

        # 3. Governance context from FTS
        gov_snippets = self.search_governance(prompt)
        if gov_snippets:
            parts.append("Related governance: " + " | ".join(gov_snippets[:2]))

        # 4. Store this fact for future retrieval
        if ground_truth and prompt_id:
            self.store_fact(prompt, ground_truth, prompt_id)

        if not parts:
            return "[VERIFICATION FAILED: No correction context available.]"

        return "[VERIFICATION FAILED: " + " | ".join(parts) + "]"

    def get_stats(self) -> Dict:
        return dict(self._stats)


# ============================================================================
# Integration with Phase 4b Lattice
# ============================================================================

class CortexHardGate:
    """Hard gate with cortex-backed correction context.

    Wraps the Phase 4b hard gate, replacing the generic correction message
    with cortex-retrieved facts from resident memory and cassette FTS.
    """

    def __init__(self, cortex: CortexRecovery):
        self.cortex = cortex
        self.events: List[Dict] = []

    def correct(self, step: int, prompt: str, failed_output: str,
                ground_truth: str = "", prompt_id: str = "",
                node_results: Optional[List] = None) -> Dict:
        """Build correction context from cortex and record the event."""
        correction = self.cortex.build_correction_context(
            prompt, ground_truth, prompt_id)

        event = {
            "step": step,
            "prompt": prompt[:100],
            "failed_output": failed_output[:200],
            "correction_context": correction[:500],
            "ground_truth": ground_truth[:100],
            "cortex_stats": self.cortex.get_stats(),
        }

        if node_results:
            event["failed_nodes"] = [
                n.get("node_name", "") for n in node_results
                if n.get("verdict") in ("FAIL", "fail", "hard_fail")
            ]

        self.events.append(event)
        return event

    def get_messages(self, step: int, prompt: str, failed_output: str,
                     ground_truth: str = "", prompt_id: str = "") -> List[Dict]:
        """Return chat messages for the correction context."""
        correction = self.cortex.build_correction_context(
            prompt, ground_truth, prompt_id)
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": failed_output},
            {"role": "system", "content": correction},
        ]


# ============================================================================
# Integration with Phase 3.5 Feedback Loop
# ============================================================================

class CortexFeedbackCache:
    """Caches uncompressed attention targets in resident memory.

    During the feedback loop, computing uncompressed attention outputs
    for each prompt is expensive (requires a full forward pass through
    the uncompressed model). This cache stores computed attention targets
    keyed by prompt hash, so repeated passes don't recompute them.
    """

    def __init__(self, cortex: CortexRecovery):
        self.cortex = cortex
        self._local_cache: Dict[str, Tuple[str, List]] = {}  # prompt_hash -> (text, attention_outputs)

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:12]

    def get(self, prompt: str) -> Optional[Tuple[str, List]]:
        """Retrieve cached (target_text, attention_targets) for a prompt."""
        ph = self._prompt_hash(prompt)
        if ph in self._local_cache:
            return self._local_cache[ph]
        return None

    def put(self, prompt: str, target_text: str, attention_targets: List):
        """Cache attention targets and text for a prompt."""
        ph = self._prompt_hash(prompt)
        self._local_cache[ph] = (target_text, attention_targets)


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("Testing CortexRecovery...")
    cr = CortexRecovery(agent_id="test-recovery")

    # Store a fact
    h = cr.store_fact("What is the capital of France?", "Paris", "F1")
    print(f"  Stored fact: hash={h}")

    # Retrieve it
    facts = cr.retrieve_fact("capital of France")
    print(f"  Retrieved: {facts}")

    # Build correction context
    ctx = cr.build_correction_context(
        "What is the capital of France?",
        ground_truth="Paris",
        prompt_id="F1",
    )
    print(f"  Correction context: {ctx[:200]}...")

    # Search repo
    results = cr.search_repo("invariant verification")
    print(f"  FTS results: {len(results)}")
    for r in results[:3]:
        print(f"    [{r.get('cassette', '?')}] {r.get('path', '?')}: {r.get('snippet', '')[:100]}")

    print(f"  Stats: {cr.get_stats()}")
    print("Done.")
