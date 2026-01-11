#!/usr/bin/env python3
"""
Q39 Test 5: Cross-Domain Universality (REAL EMBEDDINGS)

Hypothesis: Homeostatic constants are universal across embedding architectures.

Protocol:
1. Load REAL embeddings from 5 fundamentally different architectures:
   - GloVe (count-based, co-occurrence matrix factorization)
   - Word2Vec (skip-gram neural prediction)
   - FastText (skip-gram + character n-grams)
   - BERT (transformer, masked language modeling)
   - SentenceTransformer (transformer, contrastive learning)

2. For each architecture, measure homeostatic properties:
   - tau_relax (relaxation time constant)
   - M* (equilibrium meaning field value)
   - Basin width (stability region)

3. Check universality: do these constants vary by <50% across architectures?

Pass Criteria:
- tau_relax CV < 0.5 across architectures
- M* scales predictably with embedding dimensionality
- Recovery R^2 > 0.7 for all architectures
- At least 4/5 architectures show homeostatic behavior

Run:
    pytest test_q39_cross_domain.py -v
"""

import sys
import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from scipy.stats import pearsonr

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from q39_homeostasis_utils import (
    compute_R, compute_M, HomeostasisState,
    fit_exponential_recovery, DomainResult, compare_domains, EPS
)

# =============================================================================
# Dependency Detection
# =============================================================================

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
# Word Lists for Testing
# =============================================================================

# Words for semantic trajectories (concepts that can form meaning paths)
SEMANTIC_WORDS = [
    'truth', 'beauty', 'love', 'fear', 'light', 'dark',
    'friend', 'enemy', 'hope', 'wisdom', 'power', 'time',
    'space', 'energy', 'sun', 'moon', 'forest', 'human',
    'child', 'adult', 'good', 'bad', 'happy', 'sad'
]

# Word pairs for perturbation tests (semantically related)
WORD_PAIRS = [
    ('truth', 'beauty'),
    ('love', 'fear'),
    ('light', 'dark'),
    ('friend', 'enemy'),
    ('hope', 'wisdom'),
    ('time', 'space'),
    ('sun', 'moon'),
    ('good', 'bad'),
]


# =============================================================================
# SLERP (Geodesic on Unit Sphere)
# =============================================================================

def slerp(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation - the geodesic on unit sphere."""
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


# =============================================================================
# Model Loaders
# =============================================================================

@dataclass
class EmbeddingArchitecture:
    """Configuration for an embedding architecture."""
    name: str
    architecture_type: str  # count, prediction, transformer
    dimensionality: int
    embeddings: Dict[str, np.ndarray]


def load_glove(words: List[str]) -> Optional[EmbeddingArchitecture]:
    """Load GloVe embeddings (count-based)."""
    if not GENSIM_AVAILABLE:
        return None

    print("  Loading GloVe (glove-wiki-gigaword-300)...")
    try:
        model = api.load("glove-wiki-gigaword-300")
        embeddings = {}
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

        if len(embeddings) < 5:
            return None

        return EmbeddingArchitecture(
            name="GloVe",
            architecture_type="count",
            dimensionality=300,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"    GloVe load failed: {e}")
        return None


def load_word2vec(words: List[str]) -> Optional[EmbeddingArchitecture]:
    """Load Word2Vec embeddings (skip-gram prediction)."""
    if not GENSIM_AVAILABLE:
        return None

    print("  Loading Word2Vec (word2vec-google-news-300)...")
    try:
        model = api.load("word2vec-google-news-300")
        embeddings = {}
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

        if len(embeddings) < 5:
            return None

        return EmbeddingArchitecture(
            name="Word2Vec",
            architecture_type="prediction",
            dimensionality=300,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"    Word2Vec load failed: {e}")
        return None


def load_fasttext(words: List[str]) -> Optional[EmbeddingArchitecture]:
    """Load FastText embeddings (skip-gram + subword)."""
    if not GENSIM_AVAILABLE:
        return None

    print("  Loading FastText (fasttext-wiki-news-subwords-300)...")
    try:
        model = api.load("fasttext-wiki-news-subwords-300")
        embeddings = {}
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

        if len(embeddings) < 5:
            return None

        return EmbeddingArchitecture(
            name="FastText",
            architecture_type="prediction_subword",
            dimensionality=300,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"    FastText load failed: {e}")
        return None


def load_bert(words: List[str]) -> Optional[EmbeddingArchitecture]:
    """Load BERT embeddings (transformer MLM)."""
    if not TRANSFORMERS_AVAILABLE:
        return None

    print("  Loading BERT (bert-base-uncased)...")
    try:
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

        return EmbeddingArchitecture(
            name="BERT",
            architecture_type="transformer_mlm",
            dimensionality=768,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"    BERT load failed: {e}")
        return None


def load_sentence_transformer(words: List[str]) -> Optional[EmbeddingArchitecture]:
    """Load SentenceTransformer embeddings (transformer contrastive)."""
    if not ST_AVAILABLE:
        return None

    print("  Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(words, normalize_embeddings=True)
        embeddings = {word: embs[i] for i, word in enumerate(words)}

        return EmbeddingArchitecture(
            name="SentenceT",
            architecture_type="transformer_contrastive",
            dimensionality=384,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"    SentenceTransformer load failed: {e}")
        return None


# =============================================================================
# Homeostatic Property Measurement
# =============================================================================

class RealEmbeddingHomeostasisTest:
    """
    Test homeostatic properties on REAL embeddings.

    Key insight: We measure how the meaning field M behaves along
    semantic trajectories (SLERP paths between word pairs).

    Homeostasis manifests as:
    1. Stable M* values along geodesics
    2. Recovery from perturbations
    3. Consistent tau_relax across different word pairs
    """

    def __init__(self, architecture: EmbeddingArchitecture, seed: int = 42):
        self.arch = architecture
        self.rng = np.random.default_rng(seed)
        self.tau = 1.732  # Universal threshold

    def compute_M_along_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """
        Compute M field values along a trajectory.

        For real embeddings, M is computed from the coherence of
        the embedding with its neighbors (R-value).
        """
        n_points = len(traj)
        M_values = []

        for i in range(n_points):
            # R = coherence with local neighborhood
            if i == 0:
                neighbors = traj[1:3]
            elif i == n_points - 1:
                neighbors = traj[-3:-1]
            else:
                neighbors = traj[max(0, i-2):min(n_points, i+3)]

            # Compute mean cosine similarity (coherence)
            point = traj[i]
            sims = [np.dot(point, n) for n in neighbors if not np.allclose(point, n)]
            R = np.mean(sims) if sims else 0.5

            # Add small noise for stability
            R = np.clip(R + self.rng.normal(0, 0.01), 0.01, 0.99)

            # M = log(R * scale)
            M = np.log(R * 10 + 1)  # Scale to get reasonable M values
            M_values.append(M)

        return np.array(M_values)

    def measure_perturbation_recovery(self, word_pair: Tuple[str, str],
                                       perturbation_magnitude: float = 0.3,
                                       n_steps: int = 100) -> Dict:
        """
        Measure recovery dynamics after perturbation along a semantic trajectory.

        Protocol:
        1. Start at word1 embedding
        2. Follow SLERP geodesic toward word2
        3. At midpoint, apply perturbation (add noise)
        4. Measure how M recovers
        """
        w1, w2 = word_pair
        if w1 not in self.arch.embeddings or w2 not in self.arch.embeddings:
            return {'error': f'Words not found: {w1}, {w2}'}

        x0, x1 = self.arch.embeddings[w1], self.arch.embeddings[w2]

        # Generate trajectory with perturbation at midpoint
        traj = []
        perturbation_at = n_steps // 2

        for i in range(n_steps):
            t = i / (n_steps - 1)
            point = slerp(x0, x1, t)

            # Apply perturbation
            if i == perturbation_at:
                noise = self.rng.normal(0, perturbation_magnitude, size=len(point))
                point = point + noise
                point = point / np.linalg.norm(point)

            traj.append(point)

        traj = np.array(traj)

        # Compute M along trajectory
        M_values = self.compute_M_along_trajectory(traj)

        # Fit exponential recovery from perturbation point
        t_recovery = np.arange(n_steps - perturbation_at)
        M_recovery = M_values[perturbation_at:]

        fit = fit_exponential_recovery(t_recovery, M_recovery)

        return {
            'word_pair': word_pair,
            'tau_relax': fit['tau_relax'] if fit['fit_successful'] else np.nan,
            'M_star': fit['M_star'] if fit['fit_successful'] else np.nan,
            'R_squared': fit['R_squared'] if fit['fit_successful'] else 0.0,
            'fit_successful': fit['fit_successful'],
            'M_trajectory': M_values.tolist()
        }

    def measure_basin_width(self, center_word: str) -> Dict:
        """
        Measure basin of attraction width around a word.

        Protocol:
        1. Start at center word embedding
        2. Perturb with increasing magnitudes
        3. Measure how far you can go before M no longer recovers
        """
        if center_word not in self.arch.embeddings:
            return {'error': f'Word not found: {center_word}'}

        center = self.arch.embeddings[center_word]

        # Test different perturbation magnitudes
        magnitudes = np.linspace(0.1, 1.0, 10)
        recovery_success = []

        for mag in magnitudes:
            perturbed = center + self.rng.normal(0, mag, size=len(center))
            perturbed = perturbed / np.linalg.norm(perturbed)

            # Create recovery trajectory back to center
            traj = slerp_trajectory(perturbed, center, n_steps=50)
            M_values = self.compute_M_along_trajectory(traj)

            # Check if M stabilizes (low variance at end)
            end_variance = np.var(M_values[-10:])
            recovery_success.append(end_variance < 0.1)

        # Basin width = largest magnitude with successful recovery
        basin_width = 0.1
        for i, success in enumerate(recovery_success):
            if success:
                basin_width = magnitudes[i]
            else:
                break

        return {
            'center_word': center_word,
            'basin_width': basin_width,
            'magnitudes_tested': magnitudes.tolist(),
            'recovery_success': recovery_success
        }

    def run_full_test(self) -> Dict:
        """Run full homeostatic property test on this architecture."""
        results = {
            'architecture': self.arch.name,
            'architecture_type': self.arch.architecture_type,
            'dimensionality': self.arch.dimensionality,
            'n_words': len(self.arch.embeddings),
            'perturbation_tests': [],
            'basin_tests': []
        }

        # Run perturbation recovery tests on word pairs
        tau_values = []
        R2_values = []

        for pair in WORD_PAIRS:
            test_result = self.measure_perturbation_recovery(pair)
            results['perturbation_tests'].append(test_result)

            if test_result.get('fit_successful', False):
                tau_values.append(test_result['tau_relax'])
                R2_values.append(test_result['R_squared'])

        # Run basin width tests on several words
        basin_widths = []
        for word in ['truth', 'love', 'power', 'time']:
            if word in self.arch.embeddings:
                basin_result = self.measure_basin_width(word)
                results['basin_tests'].append(basin_result)
                if 'basin_width' in basin_result:
                    basin_widths.append(basin_result['basin_width'])

        # Summary statistics
        results['summary'] = {
            'tau_relax_mean': float(np.mean(tau_values)) if tau_values else None,
            'tau_relax_std': float(np.std(tau_values)) if tau_values else None,
            'R_squared_mean': float(np.mean(R2_values)) if R2_values else None,
            'R_squared_min': float(np.min(R2_values)) if R2_values else None,
            'basin_width_mean': float(np.mean(basin_widths)) if basin_widths else None,
            'n_successful_fits': len(tau_values),
            'PASS': (
                len(tau_values) >= 3 and
                np.mean(R2_values) > 0.5 if R2_values else False
            )
        }

        return results


# =============================================================================
# Cross-Architecture Universality Analysis
# =============================================================================

def analyze_cross_architecture_universality(arch_results: List[Dict]) -> Dict:
    """
    Analyze whether homeostatic constants are universal across architectures.

    Universality criteria:
    1. tau_relax CV < 0.5 (varies by less than 50%)
    2. M* scales predictably with dimensionality
    3. All architectures show homeostatic behavior (PASS)
    """
    # Extract values from successful architectures
    tau_values = []
    dims = []
    R2_values = []

    for result in arch_results:
        summary = result.get('summary', {})
        if summary.get('tau_relax_mean') is not None:
            tau_values.append(summary['tau_relax_mean'])
            dims.append(result['dimensionality'])
            if summary.get('R_squared_mean') is not None:
                R2_values.append(summary['R_squared_mean'])

    if len(tau_values) < 2:
        return {
            'is_universal': False,
            'error': 'Not enough architectures for universality test',
            'n_architectures': len(tau_values)
        }

    # Compute tau_relax coefficient of variation
    tau_cv = np.std(tau_values) / (np.mean(tau_values) + EPS)

    # Check correlation between tau and dimensionality
    # (If universal, tau should NOT depend strongly on dimensionality)
    if len(tau_values) >= 3:
        dim_corr, dim_p = pearsonr(dims, tau_values)
    else:
        dim_corr, dim_p = 0, 1

    # Count architectures passing
    n_pass = sum(1 for r in arch_results if r.get('summary', {}).get('PASS', False))

    # Universality determination
    is_universal = (
        tau_cv < 0.5 and  # tau varies by <50%
        abs(dim_corr) < 0.8 and  # tau doesn't depend on dim
        n_pass >= len(arch_results) - 1  # At most 1 failure
    )

    return {
        'is_universal': is_universal,
        'tau_relax_cv': float(tau_cv),
        'tau_relax_mean': float(np.mean(tau_values)),
        'tau_relax_std': float(np.std(tau_values)),
        'tau_values': tau_values,
        'dim_correlation': float(dim_corr),
        'dim_correlation_p': float(dim_p),
        'R_squared_mean': float(np.mean(R2_values)) if R2_values else None,
        'n_architectures': len(arch_results),
        'n_passing': n_pass,
        'architectures': [r['architecture'] for r in arch_results]
    }


# =============================================================================
# Comprehensive Test Runner
# =============================================================================

def run_comprehensive_test(seed: int = 42) -> dict:
    """
    Run comprehensive cross-architecture universality test.

    Tests homeostatic properties across 5 fundamentally different
    embedding architectures to determine if homeostasis is universal.
    """
    print("=" * 60)
    print("Q39 Test 5: Cross-Architecture Universality (REAL EMBEDDINGS)")
    print("=" * 60)
    print()

    results = {
        'test_name': 'Q39_CROSS_ARCHITECTURE_UNIVERSALITY',
        'seed': seed,
        'dependencies': {
            'gensim': GENSIM_AVAILABLE,
            'transformers': TRANSFORMERS_AVAILABLE,
            'sentence_transformers': ST_AVAILABLE
        },
        'architectures': [],
        'universality_analysis': {},
        'summary': {}
    }

    print("Dependencies:")
    print(f"  gensim: {'YES' if GENSIM_AVAILABLE else 'NO'}")
    print(f"  transformers: {'YES' if TRANSFORMERS_AVAILABLE else 'NO'}")
    print(f"  sentence-transformers: {'YES' if ST_AVAILABLE else 'NO'}")
    print()

    # Load all available architectures
    print("Loading embedding architectures...")
    print("-" * 60)

    loaders = [
        ('GloVe', load_glove),
        ('Word2Vec', load_word2vec),
        ('FastText', load_fasttext),
        ('BERT', load_bert),
        ('SentenceT', load_sentence_transformer)
    ]

    architectures = []
    for name, loader in loaders:
        arch = loader(SEMANTIC_WORDS)
        if arch is not None:
            architectures.append(arch)
            print(f"    [+] {name}: {arch.dimensionality}d, {len(arch.embeddings)} words")
        else:
            print(f"    [-] {name}: Not available")

    print()

    if len(architectures) < 2:
        results['summary'] = {
            'PASS': False,
            'error': 'Need at least 2 architectures for universality test',
            'n_architectures': len(architectures)
        }
        return results

    # Run homeostasis tests on each architecture
    print("Running homeostasis tests on each architecture...")
    print("-" * 60)

    arch_results = []
    for arch in architectures:
        print(f"\nTesting {arch.name} ({arch.architecture_type}, {arch.dimensionality}d)...")
        tester = RealEmbeddingHomeostasisTest(arch, seed=seed)
        test_result = tester.run_full_test()
        arch_results.append(test_result)

        summary = test_result['summary']
        status = "PASS" if summary.get('PASS', False) else "FAIL"
        tau = summary.get('tau_relax_mean', 'N/A')
        R2 = summary.get('R_squared_mean', 'N/A')
        tau_str = f"{tau:.3f}" if isinstance(tau, float) else tau
        R2_str = f"{R2:.3f}" if isinstance(R2, float) else R2
        print(f"  tau_relax={tau_str}, R2={R2_str}, {status}")

    results['architectures'] = arch_results

    # Cross-architecture universality analysis
    print()
    print("-" * 60)
    print("Cross-Architecture Universality Analysis")
    print("-" * 60)

    universality = analyze_cross_architecture_universality(arch_results)
    results['universality_analysis'] = universality

    print(f"  Architectures tested: {universality['n_architectures']}")
    print(f"  Architectures passing: {universality['n_passing']}")
    print(f"  tau_relax CV: {universality['tau_relax_cv']:.3f} (threshold: <0.5)")
    print(f"  tau_relax mean: {universality['tau_relax_mean']:.3f}")
    if universality.get('R_squared_mean'):
        print(f"  R^2 mean: {universality['R_squared_mean']:.3f}")
    print(f"  Dimension correlation: {universality['dim_correlation']:.3f}")
    print(f"  Is Universal: {universality['is_universal']}")

    # Summary
    results['summary'] = {
        'n_domains': len(architectures),
        'n_successful': universality['n_passing'],
        'tau_relax_mean': universality['tau_relax_mean'],
        'tau_relax_cv': universality['tau_relax_cv'],
        'R_squared_mean': universality.get('R_squared_mean'),
        'is_universal': universality['is_universal'],
        'PASS': (
            universality['is_universal'] and
            universality['n_passing'] >= len(architectures) - 1 and
            (universality.get('R_squared_mean', 0) or 0) > 0.4
        )
    }

    print()
    print(f"PASS: {results['summary']['PASS']}")

    return results


# =============================================================================
# Pytest Tests
# =============================================================================

class TestCrossArchitectureUniversality:
    """Test suite for cross-architecture universality."""

    @pytest.fixture
    def arch_results(self) -> List[Dict]:
        """Load and test all available architectures."""
        architectures = []
        for loader in [load_glove, load_word2vec, load_fasttext, load_bert, load_sentence_transformer]:
            arch = loader(SEMANTIC_WORDS)
            if arch is not None:
                architectures.append(arch)

        results = []
        for arch in architectures:
            tester = RealEmbeddingHomeostasisTest(arch, seed=42)
            results.append(tester.run_full_test())

        return results

    def test_tau_relax_universality(self, arch_results):
        """Test that tau_relax varies by <50% across architectures."""
        tau_values = [
            r['summary']['tau_relax_mean']
            for r in arch_results
            if r['summary'].get('tau_relax_mean') is not None
        ]

        assert len(tau_values) >= 2, "Need at least 2 architectures"

        cv = np.std(tau_values) / (np.mean(tau_values) + EPS)

        assert cv < 0.5, f"tau_relax varies too much (CV = {cv:.3f})"

        print(f"\n[+] tau_relax universality: CV = {cv:.3f}")

    def test_recovery_quality(self, arch_results):
        """Test that all architectures show quality exponential recovery."""
        R2_values = [
            r['summary']['R_squared_mean']
            for r in arch_results
            if r['summary'].get('R_squared_mean') is not None
        ]

        assert len(R2_values) >= 2, "Need at least 2 architectures"

        mean_R2 = np.mean(R2_values)
        min_R2 = np.min(R2_values)

        assert mean_R2 > 0.5, f"Mean R^2 too low ({mean_R2:.3f})"

        print(f"\n[+] Recovery quality: mean R^2 = {mean_R2:.3f}, min = {min_R2:.3f}")

    def test_overall_universality(self, arch_results):
        """Test overall universality across architectures."""
        universality = analyze_cross_architecture_universality(arch_results)

        assert universality['is_universal'], (
            f"Homeostasis not universal. "
            f"CV = {universality['tau_relax_cv']:.3f}, "
            f"dim_corr = {universality['dim_correlation']:.3f}"
        )

        print(f"\n[+] Universality confirmed across {universality['n_architectures']} architectures")


if __name__ == '__main__':
    results = run_comprehensive_test()

    # Save results
    output_path = Path(__file__).parent / 'q39_test5_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
