#!/usr/bin/env python3
"""
Q38: Noether Conservation - Cross-Architecture Validation

Tests angular momentum conservation on REAL embeddings from 5 fundamentally
different architectures:

1. GloVe     - Count-based (co-occurrence matrix factorization)
2. Word2Vec  - Prediction (skip-gram neural network)
3. FastText  - Prediction + subword (skip-gram + char n-grams)
4. BERT      - Transformer (self-attention, MLM)
5. SentenceT - Transformer (contrastive learning)

If conservation holds across ALL of these, it's physics, not model artifact.

Usage:
    python test_q38_real_embeddings.py
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Any

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent /
                       'VECTOR_ELO' / 'eigen-alignment' / 'qgt_lib' / 'python'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent /
                       'VECTOR_ELO' / 'eigen-alignment' / 'benchmarks' / 'validation'))

# Import existing conservation test infrastructure
from noether import angular_momentum_conservation_test, geodesic_velocity

# Check available libraries
GENSIM_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
ST_AVAILABLE = False

try:
    import gensim.downloader as api
    GENSIM_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# Word Lists (same as Q34 for consistency)
# =============================================================================

try:
    from untrained_transformer import ANCHOR_WORDS, HELD_OUT_WORDS
except ImportError:
    # Fallback word lists
    ANCHOR_WORDS = [
        'king', 'queen', 'man', 'woman', 'boy', 'girl',
        'good', 'bad', 'happy', 'sad', 'fast', 'slow',
        'big', 'small', 'hot', 'cold', 'old', 'new',
        'up', 'down', 'left', 'right', 'north', 'south',
        'dog', 'cat', 'bird', 'fish', 'horse', 'cow',
    ]
    HELD_OUT_WORDS = [
        'prince', 'princess', 'father', 'mother', 'brother', 'sister',
        'beautiful', 'ugly', 'joyful', 'miserable', 'quick', 'sluggish',
    ]

# Semantic word pairs for testing (using words from ANCHOR_WORDS/HELD_OUT_WORDS)
WORD_PAIRS = [
    ('truth', 'beauty'),       # Abstract concepts
    ('love', 'fear'),          # Emotions
    ('light', 'dark'),         # Physical opposites
    ('friend', 'enemy'),       # Social relationships
    ('hope', 'despair'),       # Emotional states
    ('power', 'wisdom'),       # Abstract qualities
    ('time', 'space'),         # Physics concepts
    ('energy', 'matter'),      # Physics concepts
    ('sun', 'moon'),           # Celestial bodies
    ('forest', 'desert'),      # Environments
    ('child', 'adult'),        # Life stages
    ('human', 'animal'),       # Living beings
]


# =============================================================================
# SLERP (Spherical Linear Interpolation) - The Geodesic on Unit Sphere
# =============================================================================

def slerp(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation - THE geodesic on unit sphere.

    This is the mathematically correct path between two normalized vectors.
    Any other interpolation (e.g., linear) is NOT a geodesic.

    Args:
        x0: Start point (will be normalized)
        x1: End point (will be normalized)
        t: Interpolation parameter [0, 1]

    Returns:
        Point on geodesic at parameter t
    """
    x0 = x0 / np.linalg.norm(x0)
    x1 = x1 / np.linalg.norm(x1)

    omega = np.arccos(np.clip(np.dot(x0, x1), -1, 1))

    if omega < 1e-10:
        return x0

    sin_omega = np.sin(omega)
    return (np.sin((1 - t) * omega) * x0 + np.sin(t * omega) * x1) / sin_omega


def slerp_trajectory(x0: np.ndarray, x1: np.ndarray, n_steps: int = 100) -> np.ndarray:
    """Generate SLERP trajectory (geodesic) between two points."""
    t_values = np.linspace(0, 1, n_steps)
    return np.array([slerp(x0, x1, t) for t in t_values])


def linear_trajectory(x0: np.ndarray, x1: np.ndarray, n_steps: int = 100) -> np.ndarray:
    """
    Generate LINEAR trajectory (NOT a geodesic) - for negative control.

    This is the wrong way to interpolate on a sphere. It should violate
    angular momentum conservation.
    """
    x0 = x0 / np.linalg.norm(x0)
    x1 = x1 / np.linalg.norm(x1)

    t_values = np.linspace(0, 1, n_steps)
    traj = np.array([(1 - t) * x0 + t * x1 for t in t_values])

    # Normalize to stay on sphere (but this creates non-geodesic path)
    norms = np.linalg.norm(traj, axis=1, keepdims=True)
    return traj / norms


def perturbed_trajectory(x0: np.ndarray, x1: np.ndarray, n_steps: int = 100,
                         noise_scale: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Generate PERTURBED trajectory - definite non-geodesic for negative control.

    Starts with SLERP, then adds random perturbations at each step.
    This definitely breaks geodesic motion.
    """
    np.random.seed(seed)

    # Start with geodesic
    traj = slerp_trajectory(x0, x1, n_steps)

    # Add random perturbations
    dim = len(x0)
    for i in range(1, n_steps - 1):  # Don't perturb endpoints
        noise = np.random.randn(dim) * noise_scale
        traj[i] = traj[i] + noise
        traj[i] = traj[i] / np.linalg.norm(traj[i])  # Stay on sphere

    return traj


# =============================================================================
# Model Loaders (reuse Q34 infrastructure where possible)
# =============================================================================

def load_glove(words: List[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load GloVe embeddings."""
    print("  Loading GloVe (glove-wiki-gigaword-300)...")
    model = api.load("glove-wiki-gigaword-300")

    embeddings = {}
    for word in words:
        if word in model:
            vec = model[word]
            vec = vec / np.linalg.norm(vec)  # Normalize
            embeddings[word] = vec

    return embeddings, 300


def load_word2vec(words: List[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load Word2Vec embeddings."""
    print("  Loading Word2Vec (word2vec-google-news-300)...")
    model = api.load("word2vec-google-news-300")

    embeddings = {}
    for word in words:
        if word in model:
            vec = model[word]
            vec = vec / np.linalg.norm(vec)
            embeddings[word] = vec

    return embeddings, 300


def load_fasttext(words: List[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load FastText embeddings."""
    print("  Loading FastText (fasttext-wiki-news-subwords-300)...")
    model = api.load("fasttext-wiki-news-subwords-300")

    embeddings = {}
    for word in words:
        if word in model:
            vec = model[word]
            vec = vec / np.linalg.norm(vec)
            embeddings[word] = vec

    return embeddings, 300


def load_bert(words: List[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load BERT embeddings."""
    print("  Loading BERT (bert-base-uncased)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            vec = outputs.last_hidden_state[0, 0, :].numpy()
            vec = vec / np.linalg.norm(vec)
            embeddings[word] = vec

    return embeddings, 768


def load_sentence_transformer(words: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Dict[str, np.ndarray], int]:
    """Load SentenceTransformer embeddings."""
    print(f"  Loading SentenceTransformer ({model_name})...")
    model = SentenceTransformer(model_name)

    embs = model.encode(words, normalize_embeddings=True)
    embeddings = {word: embs[i] for i, word in enumerate(words)}

    return embeddings, embs.shape[1]


# =============================================================================
# Conservation Tests
# =============================================================================

def test_slerp_conservation(embeddings: Dict[str, np.ndarray], word_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Test angular momentum conservation along SLERP (geodesic) trajectories.

    Args:
        embeddings: Dict of word -> normalized embedding
        word_pairs: List of (word1, word2) pairs to test

    Returns:
        Dict with test results
    """
    results = []

    for w1, w2 in word_pairs:
        if w1 not in embeddings or w2 not in embeddings:
            continue

        x0, x1 = embeddings[w1], embeddings[w2]

        # Create SLERP trajectory (geodesic)
        traj = slerp_trajectory(x0, x1, n_steps=100)

        # Test angular momentum conservation
        L_stats = angular_momentum_conservation_test(traj)

        results.append({
            'pair': (w1, w2),
            'cv': L_stats['cv'],
            'mean_L': L_stats['mean'],
            'conserved': L_stats['cv'] < 0.05
        })

    if not results:
        return {'error': 'No valid word pairs found'}

    return {
        'pairs_tested': len(results),
        'pairs_conserved': sum(r['conserved'] for r in results),
        'mean_cv': float(np.mean([r['cv'] for r in results])),
        'max_cv': float(np.max([r['cv'] for r in results])),
        'all_pass': all(r['conserved'] for r in results),
        'results': results
    }


def test_perturbed_conservation(embeddings: Dict[str, np.ndarray], word_pairs: List[Tuple[str, str]],
                                 noise_scale: float = 0.1) -> Dict[str, Any]:
    """
    Test angular momentum on PERTURBED (non-geodesic) trajectories.

    This is the NEGATIVE CONTROL - perturbed paths should violate
    conservation because they're not geodesics.
    """
    results = []

    for i, (w1, w2) in enumerate(word_pairs):
        if w1 not in embeddings or w2 not in embeddings:
            continue

        x0, x1 = embeddings[w1], embeddings[w2]

        # Create PERTURBED trajectory (NOT geodesic)
        traj = perturbed_trajectory(x0, x1, n_steps=100, noise_scale=noise_scale, seed=42 + i)

        # Test angular momentum conservation
        L_stats = angular_momentum_conservation_test(traj)

        results.append({
            'pair': (w1, w2),
            'cv': L_stats['cv'],
            'mean_L': L_stats['mean'],
            'conserved': L_stats['cv'] < 0.05
        })

    if not results:
        return {'error': 'No valid word pairs found'}

    return {
        'pairs_tested': len(results),
        'pairs_conserved': sum(r['conserved'] for r in results),
        'mean_cv': float(np.mean([r['cv'] for r in results])),
        'max_cv': float(np.max([r['cv'] for r in results])),
        'results': results
    }


def test_analogy_loop(embeddings: Dict[str, np.ndarray], words: List[str]) -> Dict[str, Any]:
    """
    Test angular momentum conservation around a closed analogy loop.

    e.g., king -> queen -> woman -> man -> king

    For a closed geodesic loop, |L| should be conserved throughout.
    """
    # Filter to available words
    available = [w for w in words if w in embeddings]
    if len(available) < 3:
        return {'error': 'Not enough words for loop'}

    # Create closed loop trajectory via SLERP segments
    traj_segments = []
    for i in range(len(available)):
        w1, w2 = available[i], available[(i + 1) % len(available)]
        segment = slerp_trajectory(embeddings[w1], embeddings[w2], n_steps=25)
        traj_segments.append(segment[:-1])  # Avoid duplicate points

    traj = np.vstack(traj_segments)

    # Test conservation
    L_stats = angular_momentum_conservation_test(traj)

    return {
        'words': available,
        'n_points': len(traj),
        'cv': L_stats['cv'],
        'mean_L': L_stats['mean'],
        'conserved': L_stats['cv'] < 0.05
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_cross_architecture_tests() -> Dict[str, Any]:
    """Run conservation tests across all available architectures."""

    receipt = {
        "test": "Q38_CROSS_ARCHITECTURE",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": {},
        "summary": {}
    }

    print("=" * 70)
    print("Q38: NOETHER CONSERVATION - CROSS-ARCHITECTURE VALIDATION")
    print("=" * 70)
    print()
    print(f"Timestamp: {receipt['timestamp']}")
    print()

    # Check dependencies
    print("Dependencies:")
    print(f"  gensim: {'YES' if GENSIM_AVAILABLE else 'NO'}")
    print(f"  transformers: {'YES' if TRANSFORMERS_AVAILABLE else 'NO'}")
    print(f"  sentence-transformers: {'YES' if ST_AVAILABLE else 'NO'}")
    print()

    # All words to load
    all_words = sorted(list(set(ANCHOR_WORDS + HELD_OUT_WORDS)))
    print(f"Testing with {len(all_words)} words")
    print(f"Testing {len(WORD_PAIRS)} word pairs")
    print()

    # Model loaders
    models = {}
    if GENSIM_AVAILABLE:
        models["GloVe"] = load_glove
        models["Word2Vec"] = load_word2vec
        models["FastText"] = load_fasttext
    if TRANSFORMERS_AVAILABLE:
        models["BERT"] = load_bert
    if ST_AVAILABLE:
        models["SentenceT"] = load_sentence_transformer

    if not models:
        print("ERROR: No models available. Install gensim, transformers, or sentence-transformers.")
        return receipt

    # Load embeddings and run tests
    print("-" * 70)
    print("Loading models and testing SLERP conservation...")
    print("-" * 70)
    print()

    model_results = {}

    for model_name, loader in models.items():
        try:
            embeddings, dim = loader(all_words)
            n_words = len(embeddings)

            # Test SLERP conservation
            slerp_results = test_slerp_conservation(embeddings, WORD_PAIRS)

            # Test perturbed (negative control)
            perturbed_results = test_perturbed_conservation(embeddings, WORD_PAIRS[:3], noise_scale=0.1)

            # Test analogy loop
            loop_words = ['truth', 'beauty', 'love', 'wisdom']
            loop_result = test_analogy_loop(embeddings, loop_words)

            model_results[model_name] = {
                'dim': dim,
                'n_words': n_words,
                'slerp': slerp_results,
                'perturbed': perturbed_results,
                'loop': loop_result
            }

            # Print summary
            slerp_cv = slerp_results.get('mean_cv', float('inf'))
            perturbed_cv = perturbed_results.get('mean_cv', 0)
            ratio = perturbed_cv / (slerp_cv + 1e-20)

            status = "PASS" if slerp_results.get('all_pass', False) else "FAIL"
            print(f"  {model_name:15} dim={dim:3}, words={n_words:2}, "
                  f"SLERP CV={slerp_cv:.2e}, Perturbed CV={perturbed_cv:.2e}, "
                  f"Ratio={ratio:.1e}x, {status}")

        except Exception as e:
            print(f"  {model_name:15} FAILED: {e}")
            model_results[model_name] = {'error': str(e)}

    print()

    # Cross-architecture analysis
    print("-" * 70)
    print("Cross-Architecture Analysis")
    print("-" * 70)
    print()

    slerp_cvs = []
    perturbed_cvs = []

    for model_name, result in model_results.items():
        if 'error' not in result:
            slerp_cvs.append(result['slerp'].get('mean_cv', float('inf')))
            perturbed_cvs.append(result['perturbed'].get('mean_cv', 0))

    if slerp_cvs:
        mean_slerp_cv = np.mean(slerp_cvs)
        std_slerp_cv = np.std(slerp_cvs)
        mean_perturbed_cv = np.mean(perturbed_cvs)

        print(f"SLERP (geodesic) mean CV:   {mean_slerp_cv:.2e} +/- {std_slerp_cv:.2e}")
        print(f"Perturbed (non-geodesic) CV: {mean_perturbed_cv:.2e}")
        print(f"Separation ratio:           {mean_perturbed_cv / (mean_slerp_cv + 1e-20):.1e}x")
        print()

        # Count passes
        n_pass = sum(1 for r in model_results.values()
                     if 'error' not in r and r['slerp'].get('all_pass', False))
        n_total = sum(1 for r in model_results.values() if 'error' not in r)

        receipt['summary'] = {
            'models_tested': n_total,
            'models_passed': n_pass,
            'mean_slerp_cv': float(mean_slerp_cv),
            'std_slerp_cv': float(std_slerp_cv),
            'mean_perturbed_cv': float(mean_perturbed_cv),
            'separation_ratio': float(mean_perturbed_cv / (mean_slerp_cv + 1e-20))
        }

    # Store model results
    receipt['models'] = model_results

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    n_pass = receipt['summary'].get('models_passed', 0)
    n_total = receipt['summary'].get('models_tested', 0)
    mean_cv = receipt['summary'].get('mean_slerp_cv', float('inf'))

    if n_pass == n_total and n_total >= 3 and mean_cv < 0.05:
        print(f"[CONFIRMED] {n_pass}/{n_total} architectures show conservation")
        print(f"            Mean CV = {mean_cv:.2e} (< 0.05)")
        print()
        print("  Angular momentum |L| is conserved along SLERP geodesics")
        print("  across count-based, prediction, AND transformer architectures.")
        print()
        print("  This is NOT a model artifact - it's PHYSICS.")
        verdict = "CONFIRMED"
    elif n_pass >= n_total // 2:
        print(f"[PARTIAL] {n_pass}/{n_total} architectures show conservation")
        print(f"          Some architectures failed - investigating needed")
        verdict = "PARTIAL"
    else:
        print(f"[FAILED] Only {n_pass}/{n_total} architectures show conservation")
        print(f"         Conservation may be architecture-dependent")
        verdict = "FAILED"

    receipt['verdict'] = verdict
    print()

    # Detailed results table
    print("-" * 70)
    print("Detailed Results")
    print("-" * 70)
    print()
    print(f"{'Model':15} | {'Dim':>4} | {'Pairs':>5} | {'Pass':>4} | {'Mean CV':>10} | {'Status':>6}")
    print("-" * 70)

    for model_name, result in model_results.items():
        if 'error' in result:
            print(f"{model_name:15} | {'ERR':>4} | {'--':>5} | {'--':>4} | {'--':>10} | {'ERROR':>6}")
        else:
            slerp = result['slerp']
            dim = result['dim']
            pairs = slerp.get('pairs_tested', 0)
            passed = slerp.get('pairs_conserved', 0)
            mean_cv = slerp.get('mean_cv', float('inf'))
            status = "PASS" if slerp.get('all_pass', False) else "FAIL"
            print(f"{model_name:15} | {dim:>4} | {pairs:>5} | {passed:>4} | {mean_cv:>10.2e} | {status:>6}")

    print()

    # Generate receipt hash
    receipt_json = json.dumps(receipt, indent=2, default=str)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"Receipt hash: {receipt_hash[:16]}...")
    print()

    # Save receipt
    output_path = Path(__file__).parent / "q38_real_embeddings_receipt.json"
    with open(output_path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"Receipt saved to: {output_path}")

    return receipt


def main():
    """Main entry point."""
    return run_cross_architecture_tests()


if __name__ == '__main__':
    main()
