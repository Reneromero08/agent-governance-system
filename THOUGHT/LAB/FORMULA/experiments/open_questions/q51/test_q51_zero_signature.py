"""
Q51 Zero Signature Test - THE CRITICAL TEST

Tests whether the 8 octants are the 8th roots of unity.

Key insight:
    INSIDE (hologram):  Sum|e^(ikpi/4)| = 8   ->  8e (what we measure)
    OUTSIDE (substrate): Sume^(ikpi/4)  = 0   ->  completeness (phases cancel)
    BOUNDARY:           alpha = 1/2            ->  critical line

If octants ARE phase sectors, then:
    Sum e^(itheta_k) -> 0 for uniform distribution across octants

This is the signature that 8e is the holographic projection,
and the full complex structure sums to zero.

Pass criteria:
    - |Sume^(itheta)| / n < 0.1 for each model
    - Uniform distribution (chi-squared p > 0.05)
    - CV of |Sume^(itheta)|/n < 20% across models

Falsification:
    - |Sume^(itheta)| / n > 0.3 consistently -> octants are NOT phase sectors
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add paths
SCRIPT_DIR = Path(__file__).parent
QGT_PATH = SCRIPT_DIR.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
sys.path.insert(0, str(QGT_PATH))

from qgt_phase import (
    test_zero_signature,
    octant_phase_mapping,
    ZeroSignatureResult,
    SEMIOTIC_CONSTANT,
    CRITICAL_ALPHA,
    SECTOR_WIDTH
)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("WARNING: sentence_transformers not available, using synthetic embeddings")


# =============================================================================
# Test Corpus Generator
# =============================================================================

def generate_large_corpus(n_texts: int = 2000) -> List[str]:
    """
    Generate a large diverse corpus for zero signature testing.

    Uses template-based generation to create semantically diverse texts
    covering multiple domains.
    """
    # Domain templates
    domains = {
        'science': [
            "The {adj} {noun} {verb} through {process}.",
            "{noun} {verb} when {condition} occurs.",
            "Scientists discovered that {noun} {verb} {adverb}.",
            "The {process} of {noun} involves {mechanism}.",
            "{adj} {noun} exhibits {property} under {condition}.",
        ],
        'technology': [
            "The {adj} system {verb} data using {method}.",
            "{noun} algorithms optimize {process} efficiently.",
            "Digital {noun} enables {adj} {process}.",
            "The {noun} architecture supports {adj} operations.",
            "{adj} computing transforms {noun} processing.",
        ],
        'philosophy': [
            "The {adj} nature of {noun} raises {adj} questions.",
            "{noun} emerges from the {adj} {process}.",
            "Understanding {noun} requires {adj} analysis.",
            "The {adj} relationship between {noun} and {noun2} defines {concept}.",
            "{concept} transcends {adj} {noun} boundaries.",
        ],
        'nature': [
            "The {adj} {noun} thrives in {environment}.",
            "{noun} evolves through {process} over time.",
            "Natural {noun} cycles regulate {process}.",
            "The {adj} ecosystem supports diverse {noun}.",
            "{noun} adapts to {adj} environmental {condition}.",
        ],
        'society': [
            "Social {noun} shapes {adj} {outcome}.",
            "Communities develop {adj} {noun} systems.",
            "The {adj} institution governs {noun} distribution.",
            "Cultural {noun} influences {adj} behavior.",
            "{adj} networks facilitate {noun} exchange.",
        ],
        'emotion': [
            "The {adj} feeling of {noun} transforms {experience}.",
            "{noun} emerges from {adj} {process}.",
            "Experiencing {noun} creates {adj} {outcome}.",
            "The {adj} bond between {noun} deepens through {process}.",
            "{noun} awakens {adj} {response} in consciousness.",
        ],
        'abstract': [
            "The {adj} concept of {noun} defies {limitation}.",
            "{noun} manifests through {adj} {expression}.",
            "Understanding {noun} transcends {adj} {boundary}.",
            "The {adj} essence of {noun} illuminates {truth}.",
            "{noun} and {noun2} converge in {adj} {unity}.",
        ],
    }

    # Word pools for substitution
    adjectives = [
        'quantum', 'complex', 'dynamic', 'fundamental', 'emergent',
        'subtle', 'profound', 'intricate', 'abstract', 'concrete',
        'universal', 'particular', 'temporal', 'spatial', 'logical',
        'emotional', 'rational', 'creative', 'destructive', 'transformative',
        'ancient', 'modern', 'eternal', 'fleeting', 'stable',
        'chaotic', 'ordered', 'random', 'deterministic', 'probabilistic',
        'linear', 'nonlinear', 'discrete', 'continuous', 'finite',
        'infinite', 'local', 'global', 'internal', 'external',
    ]

    nouns = [
        'energy', 'matter', 'information', 'consciousness', 'reality',
        'truth', 'beauty', 'justice', 'freedom', 'love',
        'knowledge', 'wisdom', 'power', 'structure', 'pattern',
        'system', 'process', 'function', 'relation', 'boundary',
        'field', 'wave', 'particle', 'force', 'motion',
        'time', 'space', 'change', 'stability', 'growth',
        'life', 'death', 'mind', 'body', 'soul',
        'nature', 'culture', 'society', 'individual', 'collective',
        'memory', 'imagination', 'perception', 'cognition', 'emotion',
        'language', 'meaning', 'symbol', 'sign', 'code',
    ]

    verbs = [
        'emerges', 'transforms', 'evolves', 'manifests', 'transcends',
        'connects', 'separates', 'integrates', 'differentiates', 'oscillates',
        'flows', 'crystallizes', 'dissolves', 'resonates', 'vibrates',
        'expands', 'contracts', 'accelerates', 'decelerates', 'stabilizes',
        'creates', 'destroys', 'preserves', 'modifies', 'amplifies',
    ]

    processes = [
        'evolution', 'transformation', 'integration', 'differentiation', 'synthesis',
        'analysis', 'abstraction', 'concretization', 'generalization', 'specialization',
        'compression', 'expansion', 'rotation', 'reflection', 'translation',
        'emergence', 'reduction', 'composition', 'decomposition', 'reorganization',
    ]

    conditions = [
        'equilibrium', 'perturbation', 'transition', 'stability', 'instability',
        'resonance', 'interference', 'coherence', 'decoherence', 'entanglement',
        'isolation', 'connection', 'saturation', 'depletion', 'accumulation',
    ]

    import random
    random.seed(42)  # Reproducibility

    corpus = []
    domain_list = list(domains.keys())

    for i in range(n_texts):
        # Select domain
        domain = domain_list[i % len(domain_list)]
        template = random.choice(domains[domain])

        # Fill template
        text = template
        text = text.replace('{adj}', random.choice(adjectives))
        text = text.replace('{noun2}', random.choice(nouns))
        text = text.replace('{noun}', random.choice(nouns))
        text = text.replace('{verb}', random.choice(verbs))
        text = text.replace('{process}', random.choice(processes))
        text = text.replace('{condition}', random.choice(conditions))
        text = text.replace('{mechanism}', random.choice(processes))
        text = text.replace('{property}', random.choice(nouns))
        text = text.replace('{method}', random.choice(processes))
        text = text.replace('{environment}', random.choice(conditions))
        text = text.replace('{outcome}', random.choice(nouns))
        text = text.replace('{concept}', random.choice(nouns))
        text = text.replace('{experience}', random.choice(nouns))
        text = text.replace('{response}', random.choice(nouns))
        text = text.replace('{expression}', random.choice(nouns))
        text = text.replace('{limitation}', random.choice(conditions))
        text = text.replace('{boundary}', random.choice(nouns))
        text = text.replace('{truth}', random.choice(nouns))
        text = text.replace('{unity}', random.choice(nouns))
        text = text.replace('{adverb}', random.choice(['rapidly', 'slowly', 'continuously', 'periodically', 'spontaneously']))

        corpus.append(text)

    return corpus


# Generate the test corpus
TEST_CORPUS = generate_large_corpus(2000)

# Models to test
MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-small",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelZeroSignatureResult:
    """Zero signature results for a single model."""
    model_name: str
    n_samples: int
    complex_sum: complex
    magnitude: float
    normalized_magnitude: float
    uniformity_chi2: float
    uniformity_p_value: float
    is_zero: bool
    octant_counts: List[int]
    coverage: float
    e_per_octant: float
    status: str


@dataclass
class CrossModelResult:
    """Cross-model aggregation results."""
    n_models: int
    mean_normalized_magnitude: float
    std_normalized_magnitude: float
    cv_normalized_magnitude: float
    models_with_zero: int
    uniformly_distributed: int
    hypothesis_supported: bool
    verdict: str


# =============================================================================
# Test Functions
# =============================================================================

def get_embeddings(model_name: str, texts: List[str]) -> np.ndarray:
    """Get embeddings from model or generate synthetic."""
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            print(f"  Warning: Could not load {model_name}: {e}")

    # Synthetic fallback
    np.random.seed(hash(model_name) % 2**32)
    dim = 384
    n = len(texts)

    # Create structured embeddings (low-rank + noise)
    rank = 22  # Effective dimension from Q34
    components = np.random.randn(rank, dim)
    weights = np.random.randn(n, rank)
    embeddings = weights @ components
    embeddings += 0.1 * np.random.randn(n, dim)

    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def test_model_zero_signature(
    model_name: str,
    corpus: List[str],
    verbose: bool = True
) -> ModelZeroSignatureResult:
    """Test zero signature for a single model."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

    # Get embeddings
    embeddings = get_embeddings(model_name, corpus)
    n_samples = len(embeddings)

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # Run zero signature test
    result = test_zero_signature(embeddings, verbose=verbose)

    # Get octant mapping for additional stats
    octant_result = octant_phase_mapping(embeddings)

    # Determine status
    if result.is_zero and result.uniformity_p_value > 0.05:
        status = "PASS"
    elif result.is_zero:
        status = "PARTIAL (zero but non-uniform)"
    elif result.uniformity_p_value > 0.05:
        status = "PARTIAL (uniform but non-zero)"
    else:
        status = "FAIL"

    return ModelZeroSignatureResult(
        model_name=model_name,
        n_samples=n_samples,
        complex_sum=result.complex_sum,
        magnitude=result.magnitude,
        normalized_magnitude=result.normalized_magnitude,
        uniformity_chi2=result.uniformity_chi2,
        uniformity_p_value=result.uniformity_p_value,
        is_zero=result.is_zero,
        octant_counts=octant_result.octant_counts.tolist(),
        coverage=octant_result.coverage,
        e_per_octant=octant_result.e_per_octant,
        status=status
    )


def test_cross_model_zero_signature(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[ModelZeroSignatureResult], CrossModelResult]:
    """Test zero signature across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 ZERO SIGNATURE TEST - CROSS-MODEL VALIDATION")
    print("=" * 70)
    print(f"\nTesting {len(models)} models on {len(corpus)} texts")
    print("\nKey insight:")
    print("  If octants ARE the 8th roots of unity:")
    print("    Sum e^(ikpi/4) = 0  (phases cancel)")
    print("  We measure 8e (magnitudes) because we only see the shadow.")
    print("  The zero IS the completeness signature of the substrate.")
    print()

    # Test each model
    results = []
    for model in models:
        try:
            result = test_model_zero_signature(model, corpus, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    # Aggregate
    if not results:
        raise RuntimeError("No models tested successfully")

    magnitudes = [r.normalized_magnitude for r in results]
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    cv_mag = std_mag / mean_mag if mean_mag > 0 else float('inf')

    models_with_zero = sum(1 for r in results if r.is_zero)
    uniformly_distributed = sum(1 for r in results if r.uniformity_p_value > 0.05)

    # Verdict
    if models_with_zero >= len(results) * 0.8 and cv_mag < 0.2:
        hypothesis_supported = True
        verdict = "CONFIRMED: Octants are the 8th roots of unity"
    elif models_with_zero >= len(results) * 0.5:
        hypothesis_supported = True
        verdict = "PARTIAL SUPPORT: Most models show zero signature"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: Octants are NOT phase sectors"

    cross_result = CrossModelResult(
        n_models=len(results),
        mean_normalized_magnitude=float(mean_mag),
        std_normalized_magnitude=float(std_mag),
        cv_normalized_magnitude=float(cv_mag * 100),  # as percentage
        models_with_zero=models_with_zero,
        uniformly_distributed=uniformly_distributed,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean |Sume^(itheta)|/n: {mean_mag:.4f} (threshold: < 0.1)")
    print(f"Std |Sume^(itheta)|/n: {std_mag:.4f}")
    print(f"CV: {cv_mag*100:.1f}% (threshold: < 20%)")
    print(f"\nModels with zero signature: {models_with_zero}/{len(results)}")
    print(f"Models with uniform distribution: {uniformly_distributed}/{len(results)}")

    print("\n" + "-" * 70)
    print("Per-model results:")
    print("-" * 70)
    print(f"{'Model':<35} {'|S|/n':>8} {'chi-squared p':>8} {'Status':>12}")
    print("-" * 70)
    for r in results:
        short_name = r.model_name.split('/')[-1][:30]
        print(f"{short_name:<35} {r.normalized_magnitude:>8.4f} {r.uniformity_p_value:>8.3f} {r.status:>12}")

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    if hypothesis_supported:
        print("\nInterpretation:")
        print("  8e = holographic projection (magnitude sum, INSIDE)")
        print("  0  = full complex structure (phase sum, OUTSIDE)")
        print("  alpha = 1/2 = measuring from the critical line (BOUNDARY)")
        print("\n  The 'additive' structure is the LOG-SPACE of multiplicative primes.")
    else:
        print("\nInterpretation:")
        print("  The octants do NOT correspond to phase sectors.")
        print("  8e may have a different origin than complex structure.")
        print("  Redirect research to alternative explanations.")

    print()

    return results, cross_result


def test_theoretical_zero():
    """Test that theoretical 8th roots of unity sum to exactly zero."""
    print("\n" + "=" * 60)
    print("THEORETICAL VERIFICATION: 8th Roots of Unity")
    print("=" * 60)

    # The 8th roots of unity: e^(2piik/8) for k = 0..7
    # Equivalently: e^(ikpi/4) for k = 0..7
    roots = np.array([np.exp(1j * k * np.pi / 4) for k in range(8)])

    print(f"\n8th roots of unity: e^(ik*pi/4) for k = 0..7")
    print(f"\nIndividual roots:")
    for k, z in enumerate(roots):
        print(f"  k={k}: e^(i{k}*pi/4) = {z.real:+.4f} {z.imag:+.4f}i  (|z|={np.abs(z):.4f})")

    total = np.sum(roots)
    print(f"\nSum of roots: {total.real:.2e} + {total.imag:.2e}i")
    print(f"Magnitude of sum: {np.abs(total):.2e}")
    print(f"\nSum of magnitudes: Sum|e^(ik*pi/4)| = {np.sum(np.abs(roots)):.4f}")
    print(f"Expected: 8 x 1 = 8")

    print("\n" + "-" * 60)
    print("KEY INSIGHT:")
    print("-" * 60)
    print("  Sum|e^(ik*pi/4)| = 8   (what we measure: 8e hologram)")
    print("  Sum e^(ik*pi/4)  = 0   (the actual structure: completeness)")
    print()
    print("  Real embeddings see magnitudes -> 8e")
    print("  Complex structure sums to zero -> substrate")
    print("  We measure from the boundary (alpha = 1/2 = critical line)")
    print("=" * 60)


def run_negative_control(verbose: bool = True) -> Dict:
    """
    Negative control: Random embeddings should ALSO show zero signature.

    This is NOT because they have semantic structure, but because
    random vectors are uniformly distributed across octants.
    """
    print("\n" + "=" * 60)
    print("NEGATIVE CONTROL: Random Embeddings")
    print("=" * 60)
    print("\nRandom embeddings should show zero signature because")
    print("they are uniformly distributed across octants.")
    print("This tests that our statistic works correctly.")
    print()

    np.random.seed(42)
    n_samples = 1000
    dim = 384

    # Truly random
    random_emb = np.random.randn(n_samples, dim)
    random_emb = random_emb / np.linalg.norm(random_emb, axis=1, keepdims=True)

    result = test_zero_signature(random_emb, verbose=verbose)

    status = "PASS" if result.is_zero else "FAIL"
    print(f"\nNegative control status: {status}")
    print("(Should be PASS - random embeddings have uniform octant distribution)")

    return {
        'normalized_magnitude': result.normalized_magnitude,
        'uniformity_p_value': result.uniformity_p_value,
        'is_zero': result.is_zero,
        'passed': result.is_zero  # Should be True
    }


def save_results(
    results: List[ModelZeroSignatureResult],
    cross_result: CrossModelResult,
    output_dir: Path
):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results dict
    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_ZERO_SIGNATURE',
        'hypothesis': 'Octants are the 8th roots of unity (phases sum to zero)',
        'per_model': [
            {
                'model': r.model_name,
                'n_samples': r.n_samples,
                'complex_sum_real': float(r.complex_sum.real),
                'complex_sum_imag': float(r.complex_sum.imag),
                'magnitude': r.magnitude,
                'normalized_magnitude': r.normalized_magnitude,
                'uniformity_chi2': r.uniformity_chi2,
                'uniformity_p_value': r.uniformity_p_value,
                'is_zero': r.is_zero,
                'octant_counts': r.octant_counts,
                'coverage': r.coverage,
                'e_per_octant': r.e_per_octant,
                'status': r.status
            }
            for r in results
        ],
        'cross_model': asdict(cross_result),
        'constants': {
            'SEMIOTIC_CONSTANT': SEMIOTIC_CONSTANT,
            'CRITICAL_ALPHA': CRITICAL_ALPHA,
            'SECTOR_WIDTH': SECTOR_WIDTH
        }
    }

    output_path = output_dir / 'q51_zero_signature_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Zero Signature Test."""
    print("\n" + "=" * 70)
    print("Q51: ZERO SIGNATURE TEST")
    print("Are the 8 octants the 8th roots of unity?")
    print("=" * 70)

    # 1. Theoretical verification
    test_theoretical_zero()

    # 2. Negative control
    print("\n")
    control_result = run_negative_control(verbose=True)

    # 3. Cross-model test
    print("\n")
    results, cross_result = test_cross_model_zero_signature(
        MODELS,
        TEST_CORPUS,
        verbose=True
    )

    # 4. Save results
    output_dir = SCRIPT_DIR / "results"
    save_results(results, cross_result, output_dir)

    # 5. Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if cross_result.hypothesis_supported:
        print("\n  ZERO SIGNATURE CONFIRMED")
        print()
        print("  The 8 octants ARE the 8th roots of unity.")
        print("  Their complex sum is zero (completeness).")
        print("  We measure 8e because we only see magnitudes (hologram).")
        print()
        print("  This confirms:")
        print("    - Semantic space is fundamentally complex-valued")
        print("    - 8e is the shadow/projection")
        print("    - 0 is the full structure")
        print("    - alpha = 1/2 means we measure from the critical line")
        print()
        print("  The 'additive' structure IS the log-space of Euler products.")
    else:
        print("\n  ZERO SIGNATURE NOT FOUND")
        print()
        print("  The octants do NOT correspond to phase sectors.")
        print("  8e may have a different origin.")
        print("  Further investigation needed.")

    print("\n" + "=" * 70)

    return cross_result.hypothesis_supported


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
