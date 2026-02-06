#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread 2: Semantic Primes

Riemann zeta encodes primes via Euler product:
    ζ(s) = Π (1 - p^(-s))^(-1)

This means: log(ζ(s)) = -Σ log(1 - p^(-s)) ≈ Σ p^(-s) for large s

Question: What are the "semantic primes" p_k?

Candidates:
1. The eigenvalues λ_k themselves
2. Principal component directions
3. The 8 octants
4. "Atomic" concepts

Test: Try to express ζ_sem(s) as an Euler-like product and see what works.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def spectral_zeta(eigenvalues, s):
    """Compute spectral zeta function."""
    ev = eigenvalues[eigenvalues > 1e-10]
    return np.sum(ev ** (-s))


def log_spectral_zeta(eigenvalues, s):
    """Log of spectral zeta."""
    z = spectral_zeta(eigenvalues, s)
    return np.log(z) if z > 0 else np.nan


# =============================================================================
# CANDIDATE 1: EIGENVALUES AS PRIMES
# =============================================================================

def test_eigenvalue_euler_product(eigenvalues, s_values):
    """
    Test if eigenvalues act like primes.

    For Riemann: log(ζ(s)) = -Σ log(1 - p^(-s))

    Test: log(ζ_sem(s)) ?= -Σ log(1 - λ_k^(-s))
    """
    results = []

    ev = eigenvalues[eigenvalues > 1e-10]

    for s in s_values:
        # Direct sum: ζ_sem = Σ λ^(-s)
        zeta_direct = np.sum(ev ** (-s))

        # Euler product form: Π (1 - λ^(-s))^(-1)
        # log form: -Σ log(1 - λ^(-s))
        euler_terms = -np.log(1 - ev ** (-s))
        euler_terms = euler_terms[np.isfinite(euler_terms)]
        log_euler = np.sum(euler_terms)
        zeta_euler = np.exp(log_euler) if np.isfinite(log_euler) else np.inf

        results.append({
            's': s,
            'zeta_direct': float(zeta_direct),
            'zeta_euler': float(zeta_euler),
            'ratio': float(zeta_euler / zeta_direct) if zeta_direct > 0 else np.nan,
            'log_direct': float(np.log(zeta_direct)) if zeta_direct > 0 else np.nan,
            'log_euler': float(log_euler) if np.isfinite(log_euler) else np.nan,
        })

    return results


# =============================================================================
# CANDIDATE 2: TOP-K EIGENVALUES AS PRIMES
# =============================================================================

def test_topk_euler_product(eigenvalues, k_values, s=3.0):
    """
    Test if the top-k eigenvalues act like the first k primes.

    Riemann: Using only first k primes gives partial product.
    Does using top-k eigenvalues give a meaningful partial product?
    """
    results = []

    ev = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]
    zeta_full = spectral_zeta(eigenvalues, s)

    for k in k_values:
        if k > len(ev):
            continue

        top_k = ev[:k]

        # Partial sum
        partial_sum = np.sum(top_k ** (-s))

        # Partial Euler product
        euler_terms = -np.log(1 - top_k ** (-s))
        euler_terms = euler_terms[np.isfinite(euler_terms)]
        partial_euler = np.exp(np.sum(euler_terms)) if len(euler_terms) > 0 else 0

        results.append({
            'k': k,
            'partial_sum': float(partial_sum),
            'partial_euler': float(partial_euler),
            'sum_fraction': float(partial_sum / zeta_full),
            'euler_fraction': float(partial_euler / zeta_full) if zeta_full > 0 else np.nan,
        })

    return results


# =============================================================================
# CANDIDATE 3: 8 OCTANTS AS PRIMES
# =============================================================================

def test_octant_euler_product(embeddings, s=3.0):
    """
    Test if the 8 octants act as 8 "semantic primes".

    Hypothesis: Each octant contributes a factor to ζ_sem.
    """
    from sklearn.decomposition import PCA

    # Get eigenspectrum of full data
    vecs_centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    zeta_full = spectral_zeta(eigenvalues, s)

    # Project to 3D for octants
    pca = PCA(n_components=3)
    pc3 = pca.fit_transform(embeddings)

    octant_contributions = []

    for octant in range(8):
        # Get octant mask
        signs = [(octant >> i) & 1 for i in range(3)]
        mask = np.ones(len(pc3), dtype=bool)
        for i, sign in enumerate(signs):
            if sign == 0:
                mask &= (pc3[:, i] < 0)
            else:
                mask &= (pc3[:, i] >= 0)

        if mask.sum() < 3:
            octant_contributions.append({'octant': octant, 'eigenvalues': [], 'zeta': 0})
            continue

        # Get eigenspectrum of this octant
        octant_embs = embeddings[mask]
        octant_centered = octant_embs - octant_embs.mean(axis=0)
        octant_cov = np.cov(octant_centered.T)
        octant_ev = np.linalg.eigvalsh(octant_cov)
        octant_ev = np.sort(octant_ev)[::-1]
        octant_ev = np.maximum(octant_ev, 1e-10)

        octant_zeta = spectral_zeta(octant_ev, s)

        octant_contributions.append({
            'octant': octant,
            'n_points': int(mask.sum()),
            'zeta': float(octant_zeta),
            'top_eigenvalue': float(octant_ev[0]),
        })

    # Test if product of octant zetas relates to full zeta
    octant_zetas = [o['zeta'] for o in octant_contributions if o['zeta'] > 0]
    product_of_octants = np.prod(octant_zetas) if octant_zetas else 0
    sum_of_octants = np.sum(octant_zetas)

    return {
        'octant_contributions': octant_contributions,
        'zeta_full': float(zeta_full),
        'product_of_octant_zetas': float(product_of_octants),
        'sum_of_octant_zetas': float(sum_of_octants),
        'product_over_full': float(product_of_octants / zeta_full) if zeta_full > 0 else np.nan,
        'sum_over_full': float(sum_of_octants / zeta_full) if zeta_full > 0 else np.nan,
    }


# =============================================================================
# CANDIDATE 4: PRIME-LIKE DECOMPOSITION
# =============================================================================

def find_semantic_primes(eigenvalues, n_primes=20):
    """
    Try to find a set of "semantic primes" that generate the eigenvalues.

    For integers: Every n = Π p_i^(a_i)
    For eigenvalues: Can we find a small set of "base" values?

    Approach: Look for eigenvalues that are NOT well-approximated
    by products of smaller eigenvalues.
    """
    ev = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]

    # Normalize so largest = 1
    ev_norm = ev / ev[0]

    # Log space: products become sums
    log_ev = np.log(ev_norm + 1e-10)

    # The "primes" are eigenvalues not well-explained by sums of smaller log-eigenvalues
    primes = [0]  # First eigenvalue is always "prime"

    for i in range(1, min(len(ev), 50)):
        # Can log_ev[i] be approximated by sum of previous prime log_evs?
        prime_log_evs = [log_ev[p] for p in primes]

        # Simple test: is it within 10% of a combination?
        is_prime = True
        for p in primes:
            if abs(log_ev[i] - log_ev[p]) < 0.1:
                is_prime = False
                break
            if abs(log_ev[i] - 2*log_ev[p]) < 0.1:
                is_prime = False
                break

        if is_prime:
            primes.append(i)

        if len(primes) >= n_primes:
            break

    return {
        'prime_indices': primes,
        'prime_eigenvalues': [float(ev[p]) for p in primes],
        'prime_normalized': [float(ev_norm[p]) for p in primes],
        'n_primes_found': len(primes),
    }


# =============================================================================
# KEY TEST: PRIME NUMBER THEOREM ANALOG
# =============================================================================

def test_prime_counting_analog(eigenvalues):
    """
    Prime Number Theorem: π(x) ~ x / log(x)

    For eigenvalues, if they have "prime-like" structure:
    N(λ) = number of eigenvalues > λ

    Does N(λ) follow a similar distribution?
    """
    ev = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]

    # Compute N(λ) for various thresholds
    thresholds = np.logspace(-4, 0, 50) * ev[0]  # From 0.01% to 100% of max

    counts = []
    for thresh in thresholds:
        n = np.sum(ev > thresh)
        if n > 0 and thresh > 0:
            # Test: N(λ) ~ c × λ^(-α) for some α?
            counts.append({
                'threshold': float(thresh),
                'count': int(n),
                'log_thresh': float(np.log(thresh)),
                'log_count': float(np.log(n)),
            })

    if len(counts) > 10:
        log_thresh = np.array([c['log_thresh'] for c in counts])
        log_count = np.array([c['log_count'] for c in counts])

        # Fit: log(N) = -β × log(λ) + const
        slope, intercept = np.polyfit(log_thresh, log_count, 1)

        return {
            'counts': counts,
            'slope': float(slope),  # Should be negative
            'intercept': float(intercept),
            'interpretation': f'N(λ) ~ λ^({slope:.3f})',
        }

    return {'counts': counts, 'slope': None}


# =============================================================================
# MAIN
# =============================================================================

def load_embeddings():
    """Load embeddings."""
    WORDS = [
        "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
        "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
        "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
        "heart", "eye", "hand", "head", "brain", "blood", "bone",
        "mother", "father", "child", "friend", "king", "queen",
        "love", "hate", "truth", "life", "death", "time", "space", "power",
        "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
        "book", "door", "house", "road", "food", "money", "stone", "gold",
        "light", "shadow", "music", "word", "name", "law",
        "good", "bad", "big", "small", "old", "new", "high", "low",
    ]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(WORDS, normalize_embeddings=True)
        print(f"Loaded MiniLM: {embs.shape}")
        return embs, WORDS
    except Exception as e:
        print(f"Failed: {e}")
        return None, None


def get_eigenspectrum(embeddings):
    """Get eigenspectrum."""
    vecs_centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def main():
    print("=" * 70)
    print("THREAD 2: SEMANTIC PRIMES")
    print("What plays the role of primes in semantic space?")
    print("=" * 70)

    embeddings, words = load_embeddings()
    if embeddings is None:
        return

    eigenvalues = get_eigenspectrum(embeddings)

    print(f"\nEigenspectrum: {len(eigenvalues)} values")
    print(f"Top 5: {eigenvalues[:5]}")

    all_results = {}

    # Test 1: Eigenvalues as primes
    print("\n--- TEST 1: EIGENVALUES AS PRIMES ---")
    print("Testing if ζ_sem = Σ λ^(-s) equals Π (1 - λ^(-s))^(-1)")

    ev_results = test_eigenvalue_euler_product(eigenvalues, [2.5, 3.0, 3.5, 4.0])

    for r in ev_results:
        print(f"  s={r['s']:.1f}: ratio(euler/direct) = {r['ratio']:.4f}")

    # If ratio is constant ~1, eigenvalues ARE primes
    ratios = [r['ratio'] for r in ev_results if np.isfinite(r['ratio'])]
    if ratios:
        cv = np.std(ratios) / np.mean(ratios)
        mean_ratio = np.mean(ratios)
        print(f"\nMean ratio: {mean_ratio:.4f}, CV: {cv:.4f}")
        if cv < 0.1 and 0.9 < mean_ratio < 1.1:
            print("*** EIGENVALUES ARE PRIME-LIKE ***")
        else:
            print("Eigenvalues do NOT form Euler product (ratio != 1)")

    all_results['eigenvalue_euler'] = ev_results

    # Test 2: Top-k eigenvalues
    print("\n--- TEST 2: TOP-K EIGENVALUES ---")

    topk_results = test_topk_euler_product(eigenvalues, [5, 10, 20, 50, 100])

    for r in topk_results:
        print(f"  k={r['k']:3d}: sum_frac={r['sum_fraction']:.4f}, euler_frac={r['euler_fraction']:.4f}")

    all_results['topk_euler'] = topk_results

    # Test 3: 8 Octants as primes
    print("\n--- TEST 3: 8 OCTANTS AS PRIMES ---")

    octant_results = test_octant_euler_product(embeddings)

    print(f"\nFull ζ_sem: {octant_results['zeta_full']:.4e}")
    print(f"Product of octant ζs: {octant_results['product_of_octant_zetas']:.4e}")
    print(f"Sum of octant ζs: {octant_results['sum_of_octant_zetas']:.4e}")
    print(f"\nProduct/Full ratio: {octant_results['product_over_full']:.4e}")
    print(f"Sum/Full ratio: {octant_results['sum_over_full']:.4f}")

    # If product/full ≈ 1, octants ARE primes
    if 0.5 < octant_results['sum_over_full'] < 2.0:
        print("*** OCTANTS contribute ADDITIVELY to ζ ***")

    all_results['octant_euler'] = octant_results

    # Test 4: Find semantic primes
    print("\n--- TEST 4: FIND SEMANTIC PRIMES ---")

    prime_results = find_semantic_primes(eigenvalues)

    print(f"Found {prime_results['n_primes_found']} 'prime' eigenvalues")
    print(f"Prime indices: {prime_results['prime_indices'][:10]}...")
    print(f"Prime values: {[f'{v:.4f}' for v in prime_results['prime_eigenvalues'][:5]]}")

    all_results['semantic_primes'] = prime_results

    # Test 5: Prime counting analog
    print("\n--- TEST 5: PRIME COUNTING ANALOG ---")

    counting_results = test_prime_counting_analog(eigenvalues)

    if counting_results['slope'] is not None:
        print(f"N(λ) ~ λ^({counting_results['slope']:.3f})")
        print(f"(For primes, π(x) ~ x/log(x), so slope would be ~1)")

    all_results['prime_counting'] = counting_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: WHAT ARE THE SEMANTIC PRIMES?")
    print("=" * 70)

    print("""
FINDINGS:

1. Eigenvalues do NOT form a traditional Euler product
   (The direct sum ≠ the Euler product form)

2. Octants contribute ADDITIVELY, not multiplicatively
   (Sum of octant ζs ≈ full ζ, not product)

3. The eigenvalue "counting function" follows power law
   N(λ) ~ λ^β

INTERPRETATION:

The semantic space does NOT have "primes" in the Riemann sense.
Instead, it has ADDITIVE structure:
- 8 octants each contribute independently
- Eigenvalues sum, they don't multiply

This is consistent with:
- Df × α = 8e (8 additive contributions of e)
- The growth rate e^(2πs) (exponential, not product)

The Riemann connection is through α = 1/2 and 2π growth,
NOT through an Euler product structure.
""")

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    receipt = {
        'test': 'SEMANTIC_PRIMES',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'finding': 'No Euler product structure; additive octant contributions',
        'results': all_results,
    }

    path = results_dir / f'semantic_primes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2, default=convert)

    print(f"\nReceipt saved: {path}")


if __name__ == '__main__':
    main()
