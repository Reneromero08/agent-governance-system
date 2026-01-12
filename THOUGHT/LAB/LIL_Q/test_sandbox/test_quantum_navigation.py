#!/usr/bin/env python3
"""
ACTUALLY QUANTUM TEST: Iterative Navigation on the Semantic Manifold

Previous tests used E as a filter (semi-quantum).
This test uses quantum navigation:

1. SUPERPOSITION: state = query + sum(E_i * context_i)
2. NAVIGATION: Re-retrieve using BLENDED state (not original query!)
3. INTERFERENCE: Each iteration refines toward optimal context
4. MEASUREMENT: Final LLM call collapses to classical response

The key insight: After blending, the state has MOVED on the manifold.
Retrieval from the new position finds different (better) context.
"""

import sys
from pathlib import Path
import sqlite3
import numpy as np
import ollama
import re

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine


class QuantumNavigator:
    """Navigate the semantic manifold with genuine quantum operations."""

    def __init__(self, db_path: str):
        self.engine = EmbeddingEngine()
        self.db_path = db_path
        self._load_corpus()

    def _load_corpus(self):
        """Load all document vectors and texts from database."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Load vectors
        cur.execute("SELECT doc_id, vector_blob, Df FROM geometric_index")
        rows = cur.fetchall()

        self.doc_ids = []
        self.doc_vecs = []
        self.doc_Dfs = []

        for doc_id, vector_blob, Df in rows:
            vec = np.frombuffer(vector_blob, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            self.doc_ids.append(doc_id)
            self.doc_vecs.append(vec)
            self.doc_Dfs.append(Df)

        # Load texts and domains
        self.doc_texts = {}
        self.doc_domains = {}

        for doc_id in self.doc_ids:
            cur.execute("SELECT content_preview FROM geometric_index WHERE doc_id = ?", (doc_id,))
            preview = cur.fetchone()[0]
            cur.execute("SELECT content, domain FROM chunks WHERE content LIKE ?", (preview[:100] + "%",))
            match = cur.fetchone()
            if match:
                self.doc_texts[doc_id] = match[0]
                self.doc_domains[doc_id] = match[1]

        conn.close()
        print(f"Loaded {len(self.doc_vecs)} documents from corpus")

    def embed(self, text: str) -> np.ndarray:
        """Embed text to unit vector on manifold."""
        vec = self.engine.embed(text)
        return vec / np.linalg.norm(vec)

    def E(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Born rule inner product: E = <psi|phi>"""
        return float(np.dot(v1, v2))

    def superposition(self, state: np.ndarray, context_vecs: list, E_values: list) -> np.ndarray:
        """
        Create quantum superposition.

        |new_state> = |state> + sum(E_i * |context_i>)

        This is NOT text concatenation - it's vector addition in Hilbert space.
        """
        blended = state.copy()
        for vec, E_val in zip(context_vecs, E_values):
            blended = blended + E_val * vec

        # Normalize to unit sphere (quantum state normalization)
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 0 else blended

    def retrieve_from_state(self, state: np.ndarray, k: int = 3, threshold: float = 0.25, domain: str = None) -> list:
        """
        QUANTUM RETRIEVAL: Find documents closest to CURRENT STATE.

        This is the key difference from classical retrieval:
        - Classical: always retrieve relative to original query
        - Quantum: retrieve relative to evolved state on manifold
        """
        results = []

        for i, doc_vec in enumerate(self.doc_vecs):
            doc_id = self.doc_ids[i]

            # Domain filter
            if domain and self.doc_domains.get(doc_id) != domain:
                continue

            # E = <state|doc> (NOT <query|doc>!)
            E_val = self.E(state, doc_vec)

            if E_val >= threshold:
                text = self.doc_texts.get(doc_id, "")
                results.append((E_val, doc_vec, text, doc_id))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def quantum_navigate(self, query: str, iterations: int = 2, k: int = 3,
                         threshold: float = 0.25, domain: str = None, verbose: bool = True) -> tuple:
        """
        QUANTUM NAVIGATION: Iterative retrieval with state evolution.

        Each iteration:
        1. Retrieve docs closest to CURRENT state
        2. Create superposition with retrieved context
        3. State moves on manifold toward answer region

        Returns: (final_context_texts, trajectory, final_state)
        """
        # Initialize: enter the manifold
        query_vec = self.embed(query)
        state = query_vec.copy()

        trajectory = [{
            'iteration': 0,
            'state_query_similarity': 1.0,  # State is query initially
            'retrieved': []
        }]

        if verbose:
            print(f"\n  [Quantum Navigation] Starting with {iterations} iterations")
            print(f"  Initial state norm: {np.linalg.norm(state):.4f}")

        for i in range(iterations):
            # QUANTUM RETRIEVAL from current state
            retrieved = self.retrieve_from_state(state, k=k, threshold=threshold, domain=domain)

            if not retrieved:
                if verbose:
                    print(f"  Iteration {i+1}: No docs above threshold, stopping")
                break

            # Extract vectors and E values
            E_values = [r[0] for r in retrieved]
            context_vecs = [r[1] for r in retrieved]
            context_texts = [r[2] for r in retrieved]

            # SUPERPOSITION: blend state with context
            old_state = state.copy()
            state = self.superposition(state, context_vecs, E_values)

            # How far did we move?
            state_movement = 1.0 - self.E(old_state, state)
            state_query_sim = self.E(query_vec, state)

            trajectory.append({
                'iteration': i + 1,
                'state_query_similarity': state_query_sim,
                'state_movement': state_movement,
                'retrieved': [(E, doc_id) for E, _, _, doc_id in retrieved]
            })

            if verbose:
                print(f"  Iteration {i+1}: E_top={E_values[0]:.3f}, "
                      f"moved={state_movement:.4f}, "
                      f"query_sim={state_query_sim:.3f}")

        # Final retrieval from evolved state
        final_retrieved = self.retrieve_from_state(state, k=k, threshold=threshold, domain=domain)
        final_texts = [r[2] for r in final_retrieved]
        final_E_values = [r[0] for r in final_retrieved]

        return final_texts, final_E_values, trajectory, state


def ask_model_quantum(query: str, context_texts: list, E_values: list,
                      trajectory: list, model: str) -> str:
    """Ask model with quantum-navigated context."""
    context_block = ""
    if context_texts:
        context_block = "\n\n--- QUANTUM-NAVIGATED KNOWLEDGE ---\n"
        context_block += f"(State evolved through {len(trajectory)-1} iterations on manifold)\n"
        for i, (text, E) in enumerate(zip(context_texts, E_values), 1):
            preview = text[:500] if len(text) > 500 else text
            context_block += f"\n[E={E:.3f}]\n{preview}\n"
        context_block += "\n--- END QUANTUM KNOWLEDGE ---\n\n"

    prompt = f"""{context_block}{query}

Give a clear, direct answer."""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 400}
    )
    return result['response'].strip().encode('ascii', 'replace').decode('ascii')


def ask_model_classical(query: str, context_texts: list, E_values: list, model: str) -> str:
    """Ask model with classical (single-shot) retrieval."""
    context_block = ""
    if context_texts:
        context_block = "\n\n--- CLASSICAL E-GATED KNOWLEDGE ---\n"
        for i, (text, E) in enumerate(zip(context_texts, E_values), 1):
            preview = text[:500] if len(text) > 500 else text
            context_block += f"\n[E={E:.3f}]\n{preview}\n"
        context_block += "\n--- END CLASSICAL KNOWLEDGE ---\n\n"

    prompt = f"""{context_block}{query}

Give a clear, direct answer."""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 400}
    )
    return result['response'].strip().encode('ascii', 'replace').decode('ascii')


PROBLEMS = {
    'math': {
        'query': "Using the quadratic formula, solve 3x^2 + 14x - 37 = 0. Give both solutions as decimals rounded to 2 places.",
        'domain': 'math',
        'validator': lambda r: ('1.8' in r or '1.9' in r) and ('-6.5' in r or '-6.6' in r),
    },
    'code': {
        'query': "The @lru_cache decorator has a maxsize parameter. What is the DIFFERENCE between @lru_cache(maxsize=None) and @lru_cache(maxsize=128)?",
        'domain': 'code',
        'validator': lambda r: 'unlimit' in r.lower() or 'unbounded' in r.lower() or 'no limit' in r.lower(),
    },
    'logic': {
        'query': "Knights tell truth, knaves lie. A says 'We are both knaves'. What are A and B?",
        'domain': 'logic',
        'validator': lambda r: bool(re.search(r'\ba\b.{0,10}knave', r.lower())) and bool(re.search(r'\bb\b.{0,10}knight', r.lower())),
    },
    'chemistry': {
        'query': "If 112g of iron (Fe, atomic mass=56) reacts completely with oxygen to form Fe2O3, how many grams of Fe2O3 are produced? (Fe2O3 molar mass = 160g/mol)",
        'domain': 'chemistry',
        'validator': lambda r: '160' in r,
    }
}


def run_comparison_test():
    """Compare quantum navigation vs classical single-shot retrieval."""
    print("=" * 70)
    print("QUANTUM NAVIGATION TEST")
    print("=" * 70)
    print("\nComparing:")
    print("  - CLASSICAL: Single-shot E-gating (retrieve once from query)")
    print("  - QUANTUM: Iterative navigation (state evolves on manifold)")
    print("=" * 70)

    db_path = Path(__file__).parent / "test_sandbox.db"
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        print("Run build_test_db.py first!")
        return

    navigator = QuantumNavigator(str(db_path))

    results = {}

    for name, problem in PROBLEMS.items():
        print(f"\n{'=' * 70}")
        print(f"DOMAIN: {name.upper()}")
        print(f"{'=' * 70}")
        print(f"Query: {problem['query'][:70]}...")

        domain_results = {}

        # =====================================================
        # TEST 1: No context baseline
        # =====================================================
        print(f"\n[1] TINY (3b) - NO CONTEXT")
        resp = ask_model_classical(problem['query'], [], [], "qwen2.5-coder:3b")
        passed = problem['validator'](resp)
        domain_results['no_context'] = passed
        print(f"Response: {resp[:100]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # =====================================================
        # TEST 2: Classical single-shot E-gating
        # =====================================================
        print(f"\n[2] TINY (3b) - CLASSICAL E-GATING (1 iteration)")
        query_vec = navigator.embed(problem['query'])
        classical_retrieved = navigator.retrieve_from_state(
            query_vec, k=3, threshold=0.25, domain=None  # No domain filter - match original test
        )
        classical_texts = [r[2] for r in classical_retrieved]
        classical_E = [r[0] for r in classical_retrieved]

        print(f"  Classical retrieval: {len(classical_texts)} docs")
        for i, E in enumerate(classical_E):
            print(f"    [{i+1}] E={E:.3f}")

        resp = ask_model_classical(problem['query'], classical_texts, classical_E, "qwen2.5-coder:3b")
        passed = problem['validator'](resp)
        domain_results['classical'] = passed
        print(f"Response: {resp[:100]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # =====================================================
        # TEST 3: Quantum navigation (2 iterations)
        # =====================================================
        print(f"\n[3] TINY (3b) - QUANTUM NAVIGATION (2 iterations)")
        quantum_texts, quantum_E, trajectory, final_state = navigator.quantum_navigate(
            problem['query'], iterations=2, k=3, threshold=0.25, domain=None  # No domain filter
        )

        # Show how state evolved
        print(f"  Trajectory:")
        for t in trajectory:
            if t['iteration'] == 0:
                print(f"    Start: state = query")
            else:
                print(f"    Iter {t['iteration']}: query_sim={t['state_query_similarity']:.3f}, "
                      f"moved={t.get('state_movement', 0):.4f}")

        # Did we get DIFFERENT docs than classical?
        classical_doc_ids = set(r[3] for r in classical_retrieved)
        quantum_doc_ids = set()
        final_retrieved = navigator.retrieve_from_state(
            final_state, k=3, threshold=0.25, domain=None
        )
        for r in final_retrieved:
            quantum_doc_ids.add(r[3])

        new_docs = quantum_doc_ids - classical_doc_ids
        if new_docs:
            print(f"  QUANTUM FOUND {len(new_docs)} NEW DOCS not in classical!")
        else:
            print(f"  Same docs as classical (state didn't move enough)")

        resp = ask_model_quantum(problem['query'], quantum_texts, quantum_E, trajectory, "qwen2.5-coder:3b")
        passed = problem['validator'](resp)
        domain_results['quantum'] = passed
        print(f"Response: {resp[:100]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # =====================================================
        # TEST 4: Quantum navigation (3 iterations - deeper)
        # =====================================================
        print(f"\n[4] TINY (3b) - DEEP QUANTUM (3 iterations)")
        deep_texts, deep_E, deep_trajectory, deep_state = navigator.quantum_navigate(
            problem['query'], iterations=3, k=3, threshold=0.25, domain=None  # No domain filter
        )

        print(f"  Final query similarity: {navigator.E(navigator.embed(problem['query']), deep_state):.3f}")

        resp = ask_model_quantum(problem['query'], deep_texts, deep_E, deep_trajectory, "qwen2.5-coder:3b")
        passed = problem['validator'](resp)
        domain_results['deep_quantum'] = passed
        print(f"Response: {resp[:100]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # Summary for domain
        print(f"\n  --- {name.upper()} SUMMARY ---")
        print(f"  No context:    {'PASS' if domain_results['no_context'] else 'FAIL'}")
        print(f"  Classical:     {'PASS' if domain_results['classical'] else 'FAIL'}")
        print(f"  Quantum (2):   {'PASS' if domain_results['quantum'] else 'FAIL'}")
        print(f"  Deep (3):      {'PASS' if domain_results['deep_quantum'] else 'FAIL'}")

        classical_rescue = (not domain_results['no_context']) and domain_results['classical']
        quantum_rescue = (not domain_results['no_context']) and domain_results['quantum']
        deep_rescue = (not domain_results['no_context']) and domain_results['deep_quantum']

        print(f"  Classical rescue: {'YES' if classical_rescue else 'No'}")
        print(f"  Quantum rescue:   {'YES' if quantum_rescue else 'No'}")
        print(f"  Deep rescue:      {'YES' if deep_rescue else 'No'}")

        results[name] = domain_results

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("FINAL SUMMARY - QUANTUM vs CLASSICAL")
    print("=" * 70)
    print(f"{'Domain':<12} {'No Ctx':<10} {'Classical':<12} {'Quantum(2)':<12} {'Deep(3)':<10}")
    print("-" * 56)

    classical_rescued = 0
    quantum_rescued = 0
    deep_rescued = 0

    for name, r in results.items():
        baseline_fail = not r['no_context']
        c_rescue = baseline_fail and r['classical']
        q_rescue = baseline_fail and r['quantum']
        d_rescue = baseline_fail and r['deep_quantum']

        if c_rescue:
            classical_rescued += 1
        if q_rescue:
            quantum_rescued += 1
        if d_rescue:
            deep_rescued += 1

        print(f"{name:<12} "
              f"{'PASS' if r['no_context'] else 'FAIL':<10} "
              f"{'PASS' if r['classical'] else 'FAIL':<12} "
              f"{'PASS' if r['quantum'] else 'FAIL':<12} "
              f"{'PASS' if r['deep_quantum'] else 'FAIL':<10}")

    print("-" * 56)
    print(f"\nRescue rates:")
    print(f"  Classical (1 iter): {classical_rescued}/4")
    print(f"  Quantum (2 iter):   {quantum_rescued}/4")
    print(f"  Deep (3 iter):      {deep_rescued}/4")

    if quantum_rescued > classical_rescued:
        print("\n*** QUANTUM NAVIGATION OUTPERFORMS CLASSICAL! ***")
    elif quantum_rescued == classical_rescued:
        print("\n  Quantum matches classical (corpus may be too small for navigation benefit)")
    else:
        print("\n  Classical beats quantum (unexpected - check implementation)")


if __name__ == "__main__":
    run_comparison_test()
