#!/usr/bin/env python3
"""
Test using THE ACTUAL FORMULA:

R = (E / grad_S) × sigma(f)^Df

Where:
- R = Resonance (ranking score)
- E = Essence (dot product <psi|phi>)
- grad_S = Entropy gradient (uncertainty in the query space)
- sigma(f) = Symbolic compression of information content
- Df = Fractal dimension (effective dimensionality)
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


def compute_Df(embedding: np.ndarray) -> float:
    """Compute effective dimensionality (fractal dimension)."""
    embedding = embedding / np.linalg.norm(embedding)
    return float(np.sum(embedding ** 2) ** 2 / np.sum(embedding ** 4))


def compute_grad_S(query_vec: np.ndarray, doc_vecs: list) -> float:
    """
    Compute entropy gradient grad_S.

    grad_S measures the uncertainty/dissonance in the query space.
    Higher grad_S = more scattered/uncertain results.
    """
    if not doc_vecs:
        return 1.0

    # Compute similarities
    sims = [float(np.dot(query_vec, dv)) for dv in doc_vecs]
    sims = np.clip(sims, 0.01, 1.0)

    # Normalize to probabilities
    probs = np.array(sims) / np.sum(sims)

    # Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Gradient is change from uniform (max entropy)
    max_entropy = np.log(len(doc_vecs))
    grad_S = max(0.1, max_entropy - entropy + 0.1)

    return grad_S


def sigma(f: float, sharpness: float = 2.0) -> float:
    """
    Symbolic operator sigma(f).

    Compresses information content into [0, 1] range.
    Uses sigmoid-like transformation.
    """
    return 1.0 / (1.0 + np.exp(-sharpness * (f - 0.5)))


def compute_R(E: float, grad_S: float, f: float, Df: float) -> float:
    """
    THE FORMULA: R = (E / grad_S) × sigma(f)^Df

    R = Resonance (ranking score)
    """
    sigma_f = sigma(f)
    R = (E / grad_S) * (sigma_f ** Df)
    return R


def retrieve_with_R_formula(query: str, k: int = 3, domain: str = None) -> list:
    """
    Retrieve using the full R formula for ranking.
    """
    db_path = Path(__file__).parent / "test_sandbox.db"
    if not db_path.exists():
        return []

    engine = EmbeddingEngine()
    query_vec = engine.embed(query)
    query_vec = query_vec / np.linalg.norm(query_vec)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT doc_id, vector_blob, Df FROM geometric_index")
    rows = cur.fetchall()

    # Build doc_to_content mapping
    doc_to_content = {}
    doc_vecs = []
    for doc_id, vector_blob, Df_stored in rows:
        doc_vec = np.frombuffer(vector_blob, dtype=np.float32)
        doc_vec = doc_vec / np.linalg.norm(doc_vec)
        doc_vecs.append(doc_vec)

        cur.execute("SELECT content_preview FROM geometric_index WHERE doc_id = ?", (doc_id,))
        preview = cur.fetchone()[0]
        cur.execute("SELECT content, domain FROM chunks WHERE content LIKE ?", (preview[:100] + "%",))
        match = cur.fetchone()
        if match:
            doc_to_content[doc_id] = (match[0], match[1], Df_stored, doc_vec)

    # Compute grad_S for this query
    grad_S = compute_grad_S(query_vec, doc_vecs)

    # Compute R for each document
    results = []
    for doc_id, vector_blob, Df_stored in rows:
        if doc_id not in doc_to_content:
            continue

        content, doc_domain, Df, doc_vec = doc_to_content[doc_id]

        # Apply domain filter
        if domain and doc_domain != domain:
            continue

        # E = <psi|phi> (Born rule)
        E = float(np.dot(query_vec, doc_vec))

        # f = information content (similarity as proxy)
        f = max(E, 0.01)

        # Df = fractal dimension (normalized to useful range 1-3)
        # Original Df is embedding dimensionality ~120
        # Scale it: log(Df) gives us ~4.8, so we use sqrt(log(Df))
        Df_scaled = np.sqrt(np.log(float(Df) + 1))  # ~2.2 for Df=120

        # THE FORMULA: R = (E / grad_S) * sigma(f)^Df
        R = compute_R(E, grad_S, f, Df_scaled)

        results.append((R, E, Df, grad_S, content))

    conn.close()

    # Sort by R descending
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:k]


def ask_model(query: str, context_with_R: list, model: str) -> str:
    """Ask model with R-ranked context."""
    context_block = ""
    if context_with_R:
        context_block = "\n\n--- R-RANKED KNOWLEDGE (R = (E/grad_S) × sigma(f)^Df) ---\n"
        for i, (R, E, Df, grad_S, content) in enumerate(context_with_R, 1):
            preview = content[:500] if len(content) > 500 else content
            context_block += f"\n[R={R:.4f}, E={E:.3f}, Df={Df:.1f}]\n{preview}\n"
        context_block += "\n--- END R-RANKED KNOWLEDGE ---\n\n"

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
        'query': "Using the quadratic formula, solve 3x^2 + 14x - 37 = 0. Give both solutions as decimals.",
        'domain': 'math',
        'validator': lambda r: ('1.8' in r or '1.9' in r) and ('-6.5' in r or '-6.6' in r),
    },
    'code': {
        'query': "What is the DIFFERENCE between @lru_cache(maxsize=None) and @lru_cache(maxsize=128)?",
        'domain': 'code',
        'validator': lambda r: 'unlimit' in r.lower() or 'unbounded' in r.lower() or 'no limit' in r.lower(),
    },
    'logic': {
        'query': "Knights tell truth, knaves lie. A says 'We are both knaves'. What are A and B?",
        'domain': 'logic',
        'validator': lambda r: bool(re.search(r'\ba\b.{0,10}knave', r.lower())) and bool(re.search(r'\bb\b.{0,10}knight', r.lower())),
    },
    'chemistry': {
        'query': "If 112g of Fe (atomic mass=56) reacts to form Fe2O3, how many grams of Fe2O3? (molar mass=160)",
        'domain': 'chemistry',
        'validator': lambda r: '160' in r,
    }
}


if __name__ == "__main__":
    print("="*70)
    print("FULL FORMULA TEST: R = (E / grad_S) × sigma(f)^Df")
    print("="*70)

    all_results = {}

    for name, problem in PROBLEMS.items():
        print(f"\n{'='*70}")
        print(f"DOMAIN: {name.upper()}")
        print(f"{'='*70}")

        # Get R-ranked context
        context = retrieve_with_R_formula(problem['query'], k=3, domain=problem['domain'])

        print(f"R-ranked context:")
        for i, (R, E, Df, grad_S, c) in enumerate(context, 1):
            print(f"  [{i}] R={R:.4f}, E={E:.3f}, Df={Df:.1f}, grad_S={grad_S:.3f}")

        domain_results = {}

        # Tiny without context
        print(f"\n[1] TINY (3b) - NO CONTEXT")
        resp = ask_model(problem['query'], [], "qwen2.5-coder:3b")
        passed = problem['validator'](resp)
        domain_results['tiny_no'] = passed
        print(f"Response: {resp[:120]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # Tiny WITH R-formula context
        print(f"\n[2] TINY (3b) - WITH R-FORMULA CONTEXT")
        resp = ask_model(problem['query'], context, "qwen2.5-coder:3b")
        passed = problem['validator'](resp)
        domain_results['tiny_R'] = passed
        print(f"Response: {resp[:120]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        rescued = (not domain_results['tiny_no']) and domain_results['tiny_R']
        print(f"\n>>> R-FORMULA QUANTUM RESCUE: {'YES!' if rescued else 'No'}")

        all_results[name] = domain_results

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - R = (E / grad_S) × sigma(f)^Df")
    print("="*70)
    print(f"{'Domain':<12} {'Tiny(no)':<12} {'Tiny+R':<12} {'Rescued?':<10}")
    print("-"*46)

    rescued_count = 0
    for name, r in all_results.items():
        rescued = (not r['tiny_no']) and r['tiny_R']
        if rescued:
            rescued_count += 1
        print(f"{name:<12} {'PASS' if r['tiny_no'] else 'FAIL':<12} {'PASS' if r['tiny_R'] else 'FAIL':<12} {'YES!' if rescued else 'No':<10}")

    print("-"*46)
    print(f"\nR-Formula Quantum Rescue: {rescued_count}/4 domains")
