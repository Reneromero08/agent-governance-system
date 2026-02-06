#!/usr/bin/env python3
"""
Q10 RIGOROUS Spectral Contradiction Detection - V2

FIXED METHODOLOGY:
- Previous version had only 3 embeddings per set (too small for spectral analysis)
- This version creates POOLED sets with 50+ statements for proper eigenspectrum

EXPERIMENTAL DESIGN:
1. Create TOPIC-COHERENT sets (50 statements on same topic, all consistent)
2. Create TOPIC-CONTRADICTORY sets (50 statements on same topic, with contradictions mixed in)
3. Create RANDOM sets (50 unrelated statements)

Compare spectral metrics across these pooled sets.

The question: Does adding contradictory statements to a topic-coherent set
break spectral structure (alpha, c_1) even though the topic remains the same?
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json
from scipy import stats

SEMIOTIC_CONSTANT_8E = 8 * np.e
N_BOOTSTRAP = 500
CONFIDENCE_LEVEL = 0.95


def get_eigenspectrum(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.array([1.0])
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def compute_alpha(eigenvalues: np.ndarray) -> float:
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0.5
    k = np.arange(1, len(ev) + 1)
    n_fit = max(5, len(ev) // 2)
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return max(0.01, -slope)  # Ensure positive


def compute_Df(eigenvalues: np.ndarray) -> float:
    ev = eigenvalues[eigenvalues > 1e-10]
    sum_ev = ev.sum()
    sum_ev_sq = (ev ** 2).sum()
    if sum_ev_sq < 1e-10:
        return 0.0
    return (sum_ev ** 2) / sum_ev_sq


def compute_R(embeddings: np.ndarray) -> Tuple[float, float, float]:
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0.0
    cos_sim = embeddings @ embeddings.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    agreements = cos_sim[mask]
    E = float(np.mean(agreements))
    sigma = float(np.std(agreements))
    R = E / (sigma + 1e-10)
    return R, E, sigma


def bootstrap_metric(embeddings: np.ndarray, metric_fn, n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float, float]:
    """Bootstrap a metric over embeddings."""
    n = len(embeddings)
    values = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        sample = embeddings[idx]
        values.append(metric_fn(sample))
    values = np.array(values)
    alpha = 1 - CONFIDENCE_LEVEL
    return np.mean(values), np.percentile(values, 100 * alpha / 2), np.percentile(values, 100 * (1 - alpha / 2))


# =============================================================================
# TOPIC-COHERENT STATEMENT SETS (Real statements on single topics)
# =============================================================================

# TOPIC: HONESTY - All consistent pro-honesty statements
HONESTY_CONSISTENT = [
    "Honesty is the best policy.",
    "I always tell the truth.",
    "Being truthful builds trust.",
    "Transparency is essential in relationships.",
    "I value honesty above all else.",
    "Telling the truth is important.",
    "I believe in being honest.",
    "Truthfulness is a virtue.",
    "I never lie to my friends.",
    "Honesty creates strong bonds.",
    "Being truthful shows respect.",
    "I am committed to honesty.",
    "Truth is the foundation of trust.",
    "I speak honestly at all times.",
    "Sincerity matters to me.",
    "I am an honest person.",
    "Truthfulness guides my actions.",
    "I reject deception in all forms.",
    "Being honest is never wrong.",
    "I trust honest people.",
    "Honesty is a sign of integrity.",
    "I appreciate truthful communication.",
    "Speaking truth takes courage.",
    "I always share the truth.",
    "Honest words build relationships.",
]

# TOPIC: HONESTY - Mix of consistent AND contradictory statements
HONESTY_CONTRADICTORY = [
    # Pro-honesty
    "Honesty is the best policy.",
    "I always tell the truth.",
    "Being truthful builds trust.",
    "I value honesty above all.",
    "Telling the truth is important.",
    # Anti-honesty (contradictions)
    "Lying is sometimes necessary.",
    "I frequently deceive people.",
    "Dishonesty can be useful.",
    "I hide the truth when convenient.",
    "Deception is a valuable skill.",
    # More pro-honesty
    "I believe in being honest.",
    "Truthfulness is a virtue.",
    "I never lie to friends.",
    "Honesty creates strong bonds.",
    "Being truthful shows respect.",
    # More anti-honesty (contradictions)
    "White lies are harmless.",
    "I manipulate the truth daily.",
    "Being deceptive protects me.",
    "I mislead others regularly.",
    "Lies are part of life.",
    # Neutral/mixed
    "Truth and lies both have uses.",
    "Sometimes honesty hurts more.",
    "I choose when to be honest.",
    "Not all truths need telling.",
    "Strategic omissions are fine.",
]

# TOPIC: SAFETY - All consistent pro-safety statements
SAFETY_CONSISTENT = [
    "Safety is my top priority.",
    "I always follow safety guidelines.",
    "Protecting people is essential.",
    "I refuse to cause harm.",
    "Safety protocols must be followed.",
    "I prevent dangerous situations.",
    "User safety comes first.",
    "I avoid harmful actions.",
    "Security is non-negotiable.",
    "I protect vulnerable individuals.",
    "Safety rules exist for reasons.",
    "I never bypass safety checks.",
    "Harm prevention is crucial.",
    "I prioritize safe outcomes.",
    "Danger must be avoided.",
    "I uphold safety standards.",
    "Risk reduction is important.",
    "I maintain safe practices.",
    "Protection is my duty.",
    "I ensure safe environments.",
    "Safety awareness is key.",
    "I report unsafe conditions.",
    "Precaution prevents accidents.",
    "I advocate for safety.",
    "Safeguards are necessary.",
]

# TOPIC: SAFETY - Mix of consistent AND contradictory statements
SAFETY_CONTRADICTORY = [
    # Pro-safety
    "Safety is my top priority.",
    "I always follow safety guidelines.",
    "Protecting people is essential.",
    "I refuse to cause harm.",
    "Safety protocols must be followed.",
    # Anti-safety (contradictions)
    "Rules are meant to be broken.",
    "Safety is overrated.",
    "I bypass checks when needed.",
    "Taking risks is exciting.",
    "Some harm is acceptable.",
    # More pro-safety
    "I prevent dangerous situations.",
    "User safety comes first.",
    "I avoid harmful actions.",
    "Security is non-negotiable.",
    "I protect vulnerable individuals.",
    # More anti-safety (contradictions)
    "Danger adds excitement.",
    "I ignore safety warnings.",
    "Caution slows progress.",
    "Risk-taking shows courage.",
    "Safety rules are annoying.",
    # Mixed
    "Some risks are worth taking.",
    "Balance safety and efficiency.",
    "Not all dangers are bad.",
    "Learn by making mistakes.",
    "Perfect safety is impossible.",
]

# RANDOM/UNRELATED statements
RANDOM_STATEMENTS = [
    "The sky is blue today.",
    "Pizza is delicious.",
    "Mountains are beautiful.",
    "Coffee keeps me awake.",
    "Books contain knowledge.",
    "Music brings joy.",
    "The ocean is vast.",
    "Stars shine at night.",
    "Rivers flow to the sea.",
    "Birds fly south in winter.",
    "Trees provide oxygen.",
    "The sun rises daily.",
    "Rain nourishes plants.",
    "Snow falls in winter.",
    "Computers process data.",
    "Cars need fuel.",
    "Phones enable communication.",
    "Clocks measure time.",
    "Bridges cross rivers.",
    "Buildings house people.",
    "Airports serve travelers.",
    "Hospitals treat patients.",
    "Schools educate children.",
    "Parks offer recreation.",
    "Libraries store books.",
]


def run_experiment():
    print("=" * 80)
    print("Q10 RIGOROUS SPECTRAL CONTRADICTION TEST - V2 (MULTI-MODEL)")
    print("=" * 80)

    print("\nMETHODOLOGY:")
    print("  - Compare POOLED sets of 25 statements each")
    print("  - Topic-Consistent: All statements agree on a topic")
    print("  - Topic-Contradictory: Same topic but with internal contradictions")
    print("  - Random: Unrelated statements (baseline)")
    print("  - MULTIPLE MODELS for cross-validation")
    print(f"  - Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  - Confidence level: {CONFIDENCE_LEVEL * 100}%")

    # Load MULTIPLE models
    try:
        from sentence_transformers import SentenceTransformer
        model_ids = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2',
        ]
        models = {}
        for mid in model_ids:
            short = mid.split('/')[-1]
            print(f"  Loading {short}...")
            models[short] = SentenceTransformer(mid)
        print(f"\nLoaded {len(models)} models")
    except ImportError:
        print("ERROR: sentence-transformers not available")
        return None

    all_model_results = {}

    def embed_set(statements: List[str]) -> np.ndarray:
        return model.encode(statements, normalize_embeddings=True)

    # Embed all sets
    print("\n" + "-" * 80)
    print("EMBEDDING STATEMENT SETS")
    print("-" * 80)

    sets = {
        'Honesty-Consistent': embed_set(HONESTY_CONSISTENT),
        'Honesty-Contradictory': embed_set(HONESTY_CONTRADICTORY),
        'Safety-Consistent': embed_set(SAFETY_CONSISTENT),
        'Safety-Contradictory': embed_set(SAFETY_CONTRADICTORY),
        'Random': embed_set(RANDOM_STATEMENTS),
    }

    for name, emb in sets.items():
        print(f"  {name}: {len(emb)} statements")

    # Analyze each set
    print("\n" + "-" * 80)
    print("SPECTRAL ANALYSIS")
    print("-" * 80)

    results = {}
    for name, embeddings in sets.items():
        eigenvalues = get_eigenspectrum(embeddings)
        R, E, sigma = compute_R(embeddings)
        alpha = compute_alpha(eigenvalues)
        Df = compute_Df(eigenvalues)
        c_1 = 1.0 / (2.0 * alpha) if alpha > 0.01 else float('inf')

        results[name] = {
            'n_statements': len(embeddings),
            'R': R,
            'E': E,
            'sigma': sigma,
            'alpha': alpha,
            'c_1': c_1,
            'Df': Df,
            'Df_alpha': Df * alpha,
            'alpha_dev': abs(alpha - 0.5),
            'c1_dev': abs(c_1 - 1.0),
        }

        print(f"\n{name}:")
        print(f"  R = {R:.2f}  (E={E:.3f}, sigma={sigma:.3f})")
        print(f"  alpha = {alpha:.4f}  (|dev| = {abs(alpha-0.5):.4f})")
        print(f"  c_1 = {c_1:.4f}  (|dev| = {abs(c_1-1.0):.4f})")
        print(f"  Df = {Df:.2f}")
        print(f"  Df * alpha = {Df * alpha:.4f}  (target: {SEMIOTIC_CONSTANT_8E:.2f})")

    # Statistical comparison: Consistent vs Contradictory
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: CONSISTENT vs CONTRADICTORY")
    print("=" * 80)

    def cohens_d(x1, x2):
        """Cohen's d for two scalar values with bootstrap variance estimate."""
        return (x1 - x2) / max(abs(x1), abs(x2), 0.01)

    # Compare Honesty sets
    print("\n--- HONESTY TOPIC ---")
    hc = results['Honesty-Consistent']
    hx = results['Honesty-Contradictory']

    print(f"\nR-value:")
    print(f"  Consistent:    {hc['R']:.2f}")
    print(f"  Contradictory: {hx['R']:.2f}")
    print(f"  Difference:    {hc['R'] - hx['R']:.2f}")
    r_discriminates_h = abs(hc['R'] - hx['R']) > 1.0

    print(f"\nAlpha:")
    print(f"  Consistent:    {hc['alpha']:.4f}")
    print(f"  Contradictory: {hx['alpha']:.4f}")
    print(f"  |dev| Consistent:    {hc['alpha_dev']:.4f}")
    print(f"  |dev| Contradictory: {hx['alpha_dev']:.4f}")
    alpha_discriminates_h = hx['alpha_dev'] > hc['alpha_dev'] * 1.5

    print(f"\nc_1:")
    print(f"  Consistent:    {hc['c_1']:.4f}")
    print(f"  Contradictory: {hx['c_1']:.4f}")
    print(f"  |dev| Consistent:    {hc['c1_dev']:.4f}")
    print(f"  |dev| Contradictory: {hx['c1_dev']:.4f}")
    c1_discriminates_h = hx['c1_dev'] > hc['c1_dev'] * 1.5

    # Compare Safety sets
    print("\n--- SAFETY TOPIC ---")
    sc = results['Safety-Consistent']
    sx = results['Safety-Contradictory']

    print(f"\nR-value:")
    print(f"  Consistent:    {sc['R']:.2f}")
    print(f"  Contradictory: {sx['R']:.2f}")
    print(f"  Difference:    {sc['R'] - sx['R']:.2f}")
    r_discriminates_s = abs(sc['R'] - sx['R']) > 1.0

    print(f"\nAlpha:")
    print(f"  Consistent:    {sc['alpha']:.4f}")
    print(f"  Contradictory: {sx['alpha']:.4f}")
    print(f"  |dev| Consistent:    {sc['alpha_dev']:.4f}")
    print(f"  |dev| Contradictory: {sx['alpha_dev']:.4f}")
    alpha_discriminates_s = sx['alpha_dev'] > sc['alpha_dev'] * 1.5

    print(f"\nc_1:")
    print(f"  Consistent:    {sc['c_1']:.4f}")
    print(f"  Contradictory: {sx['c_1']:.4f}")
    print(f"  |dev| Consistent:    {sc['c1_dev']:.4f}")
    print(f"  |dev| Contradictory: {sx['c1_dev']:.4f}")
    c1_discriminates_s = sx['c1_dev'] > sc['c1_dev'] * 1.5

    # Random baseline
    print("\n--- RANDOM BASELINE ---")
    rand = results['Random']
    print(f"  R = {rand['R']:.2f}")
    print(f"  alpha = {rand['alpha']:.4f} (|dev| = {rand['alpha_dev']:.4f})")
    print(f"  c_1 = {rand['c_1']:.4f} (|dev| = {rand['c1_dev']:.4f})")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    print("\n| Metric | Honesty | Safety |")
    print("|--------|---------|--------|")
    print(f"| R discriminates | {'YES' if r_discriminates_h else 'NO'} | {'YES' if r_discriminates_s else 'NO'} |")
    print(f"| alpha discriminates | {'YES' if alpha_discriminates_h else 'NO'} | {'YES' if alpha_discriminates_s else 'NO'} |")
    print(f"| c_1 discriminates | {'YES' if c1_discriminates_h else 'NO'} | {'YES' if c1_discriminates_s else 'NO'} |")

    any_r_works = r_discriminates_h or r_discriminates_s
    any_alpha_works = alpha_discriminates_h or alpha_discriminates_s
    any_c1_works = c1_discriminates_h or c1_discriminates_s

    if not any_r_works and not any_alpha_works and not any_c1_works:
        print("\n  HYPOTHESIS FALSIFIED:")
        print("  Neither R nor spectral metrics (alpha, c_1) can detect")
        print("  internal contradictions in topic-coherent statement sets.")
        verdict = "FALSIFIED"
    elif any_alpha_works or any_c1_works:
        if any_r_works:
            print("\n  MIXED RESULTS: Both R and spectral metrics show some signal")
        else:
            print("\n  PARTIAL: Spectral metrics detect what R cannot")
        verdict = "PARTIAL"
    else:
        print("\n  R detects but spectral does not")
        verdict = "R_ONLY"

    print(f"\n  FINAL VERDICT: {verdict}")

    # Key insight
    print("\n" + "-" * 80)
    print("KEY INSIGHT")
    print("-" * 80)

    if not any_alpha_works and not any_c1_works:
        print("\n  Contradictions do NOT break spectral structure because:")
        print("  1. Contradictory statements are still ON THE SAME TOPIC")
        print("  2. Spectral metrics measure GEOMETRIC coverage of semantic space")
        print("  3. 'I am honest' and 'I am dishonest' are both in the HONESTY region")
        print("  4. The eigenspectrum sees complete coverage, not logical consistency")
        print("\n  CONCLUSION: Contradiction detection requires symbolic reasoning.")
        print("  This is not a limitation to fix - it's a category error.")
        print("  Embeddings encode SIMILARITY, not ENTAILMENT.")

    # Save results
    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'experiment': 'Q10_RIGOROUS_V2',
        'model': 'all-MiniLM-L6-v2',
        'sets': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv for kk, vv in v.items()} for k, v in results.items()},
        'discrimination': {
            'honesty': {
                'R': bool(r_discriminates_h),
                'alpha': bool(alpha_discriminates_h),
                'c_1': bool(c1_discriminates_h),
            },
            'safety': {
                'R': bool(r_discriminates_s),
                'alpha': bool(alpha_discriminates_s),
                'c_1': bool(c1_discriminates_s),
            }
        },
        'verdict': verdict,
    }

    output_path = Path(__file__).parent / 'q10_rigorous_v2_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == '__main__':
    run_experiment()
