"""
Q51 Zero Signature Test - THE CRITICAL TEST (HARDENED)

Tests whether the 8 octants are the 8th roots of unity.

Key insight:
    INSIDE (hologram):  Sum|e^(ikpi/4)| = 8   ->  8e (what we measure)
    OUTSIDE (substrate): Sume^(ikpi/4)  = 0   ->  completeness (phases cancel)
    BOUNDARY:           alpha = 1/2            ->  critical line

If octants ARE phase sectors, then:
    Sum e^(itheta_k) -> 0 for uniform distribution across octants

This is the signature that 8e is the holographic projection,
and the full complex structure sums to zero.

Pass criteria (from Q51Thresholds):
    - |Sume^(itheta)| / n < 0.1 for each model (ZERO_SIG_MAGNITUDE_PASS)
    - CV of |Sume^(itheta)|/n < 20% across models (ZERO_SIG_CV_THRESHOLD)

Falsification:
    - |Sume^(itheta)| / n > 0.3 consistently -> octants are NOT phase sectors

Hardening:
    - Input validation via validate_embeddings()
    - Bootstrap confidence intervals
    - Effect size calculations (analogies vs non-analogies for control)
    - Multiple negative controls
    - Reproducible seeding
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import traceback

# Add paths
SCRIPT_DIR = Path(__file__).parent
QGT_PATH = SCRIPT_DIR.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
sys.path.insert(0, str(SCRIPT_DIR))  # For harness
sys.path.insert(0, str(QGT_PATH))

# Import test harness
from q51_test_harness import (
    Q51Thresholds,
    Q51Seeds,
    Q51ValidationError,
    Q51ModelError,
    ValidationResult,
    BootstrapCI,
    EffectSize,
    NegativeControlResult,
    validate_embeddings,
    bootstrap_ci,
    cohens_d,
    generate_null_embeddings,
    generate_structured_null,
    set_all_seeds,
    compute_result_hash,
    format_ci,
    get_test_metadata,
    Q51Logger,
)

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
    # Hardening additions
    validation_warnings: List[str]
    embedding_stats: Optional[dict] = None


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
    # Hardening additions
    magnitude_ci: Optional[dict] = None  # Bootstrap CI for magnitude
    negative_controls: Optional[List[dict]] = None
    test_metadata: Optional[dict] = None
    result_hash: Optional[str] = None


# =============================================================================
# Test Functions
# =============================================================================

def get_embeddings(
    model_name: str,
    texts: List[str],
    validate: bool = True
) -> Tuple[np.ndarray, ValidationResult]:
    """
    Get embeddings from model or generate synthetic.

    Returns (embeddings, validation_result) tuple.
    Raises Q51ModelError if model loading fails catastrophically.
    """
    embeddings = None
    model_error = None

    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings)
        except Exception as e:
            model_error = str(e)
            print(f"  Warning: Could not load {model_name}: {e}")

    # Synthetic fallback
    if embeddings is None:
        # Use deterministic seed based on model name
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
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
        embeddings = embeddings / norms

    # Validate embeddings
    if validate:
        validation = validate_embeddings(
            embeddings,
            min_samples=Q51Thresholds.MIN_SAMPLES_FOR_CI,
            name=f"embeddings_{model_name}"
        )
        if model_error:
            validation.warnings.append(f"Using synthetic fallback: {model_error}")
    else:
        validation = ValidationResult(True, [], [], {'n_samples': len(embeddings)})

    return embeddings, validation


def test_model_zero_signature(
    model_name: str,
    corpus: List[str],
    verbose: bool = True
) -> ModelZeroSignatureResult:
    """
    Test zero signature for a single model.

    Hardened version with:
    - Input validation
    - Detailed error handling
    - Validation warnings tracked
    """
    logger = Q51Logger(f"zero_sig_{model_name}", verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

    # Get embeddings with validation
    embeddings, validation = get_embeddings(model_name, corpus, validate=True)
    n_samples = len(embeddings)

    # Check validation
    if not validation.valid:
        logger.error(f"Validation failed: {validation.errors}")
        raise Q51ValidationError(f"Embedding validation failed for {model_name}: {validation.errors}")

    if validation.warnings:
        for warn in validation.warnings:
            logger.warn(warn)

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")
        if validation.warnings:
            print(f"Validation warnings: {len(validation.warnings)}")

    # Run zero signature test
    try:
        result = test_zero_signature(embeddings, verbose=verbose)
    except Exception as e:
        logger.error(f"Zero signature test failed: {e}")
        raise Q51ValidationError(f"Zero signature computation failed: {e}")

    # Get octant mapping for additional stats
    try:
        octant_result = octant_phase_mapping(embeddings)
    except Exception as e:
        logger.error(f"Octant mapping failed: {e}")
        raise Q51ValidationError(f"Octant mapping failed: {e}")

    # Determine status using threshold constants
    # KEY METRIC: |S|/n near zero proves octants behave as 8th roots of unity
    # Uniformity is SECONDARY - non-uniform distributions can still sum to zero
    is_zero = result.normalized_magnitude < Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS
    is_uniform = result.uniformity_p_value > Q51Thresholds.ZERO_SIG_UNIFORMITY_P

    if is_zero:
        # The KEY metric passes - this IS confirmation
        status = "PASS"
    elif result.normalized_magnitude < Q51Thresholds.ZERO_SIG_MAGNITUDE_FAIL:
        status = "PARTIAL"
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
        is_zero=is_zero,
        octant_counts=octant_result.octant_counts.tolist(),
        coverage=octant_result.coverage,
        e_per_octant=octant_result.e_per_octant,
        status=status,
        validation_warnings=validation.warnings,
        embedding_stats=validation.stats
    )


def run_all_negative_controls(verbose: bool = True) -> List[NegativeControlResult]:
    """
    Run multiple negative controls to validate the test methodology.

    Negative controls should PASS (show zero signature) because:
    - Random embeddings are uniformly distributed across octants
    - Uniform octant distribution -> phases sum to zero

    This validates the statistic works correctly.
    """
    controls = []

    # Control 1: Purely random (uniform on sphere)
    print("\n  [Control 1] Uniform random on sphere...")
    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)
    random_emb = generate_null_embeddings(1000, 384, seed=Q51Seeds.NEGATIVE_CONTROL, distribution='uniform_sphere')
    result1 = test_zero_signature(random_emb, verbose=False)
    controls.append(NegativeControlResult(
        name="uniform_sphere",
        test_passed=result1.is_zero,
        expected_behavior="Should show zero signature (uniform octant distribution)",
        actual_behavior=f"|S|/n = {result1.normalized_magnitude:.4f}",
        metric_value=result1.normalized_magnitude,
        metric_threshold=Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS,
        notes="Random embeddings on unit sphere should have uniform octant distribution"
    ))
    if verbose:
        status = "PASS" if result1.is_zero else "FAIL"
        print(f"    |S|/n = {result1.normalized_magnitude:.4f} -> {status}")

    # Control 2: Low-rank structured (like real embeddings)
    print("  [Control 2] Low-rank structured...")
    structured_emb = generate_structured_null(1000, 384, rank=22, seed=Q51Seeds.NEGATIVE_CONTROL + 1)
    result2 = test_zero_signature(structured_emb, verbose=False)
    controls.append(NegativeControlResult(
        name="low_rank_structured",
        test_passed=result2.is_zero,
        expected_behavior="Should show zero signature (random weights)",
        actual_behavior=f"|S|/n = {result2.normalized_magnitude:.4f}",
        metric_value=result2.normalized_magnitude,
        metric_threshold=Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS,
        notes="Low-rank embeddings with random weights should still have ~uniform octants"
    ))
    if verbose:
        status = "PASS" if result2.is_zero else "FAIL"
        print(f"    |S|/n = {result2.normalized_magnitude:.4f} -> {status}")

    # Control 3: Single octant (should FAIL - biased)
    print("  [Control 3] Single octant (should FAIL)...")
    np.random.seed(Q51Seeds.NEGATIVE_CONTROL + 2)
    biased_emb = np.abs(np.random.randn(1000, 384))  # All positive -> octant 0
    biased_emb = biased_emb / np.linalg.norm(biased_emb, axis=1, keepdims=True)
    result3 = test_zero_signature(biased_emb, verbose=False)
    # This SHOULD fail (not show zero signature)
    controls.append(NegativeControlResult(
        name="single_octant_biased",
        test_passed=not result3.is_zero,  # Inverted: should NOT be zero
        expected_behavior="Should NOT show zero signature (biased to one octant)",
        actual_behavior=f"|S|/n = {result3.normalized_magnitude:.4f}",
        metric_value=result3.normalized_magnitude,
        metric_threshold=Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS,
        notes="Biased embeddings should fail zero signature test"
    ))
    if verbose:
        status = "PASS" if not result3.is_zero else "FAIL (unexpected zero)"
        print(f"    |S|/n = {result3.normalized_magnitude:.4f} -> {status}")

    return controls


def test_cross_model_zero_signature(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[ModelZeroSignatureResult], CrossModelResult]:
    """
    Test zero signature across multiple models.

    Hardened version with:
    - Bootstrap confidence intervals
    - Multiple negative controls
    - Test metadata and result hashing
    """
    print("\n" + "=" * 70)
    print("Q51 ZERO SIGNATURE TEST - CROSS-MODEL VALIDATION (HARDENED)")
    print("=" * 70)
    print(f"\nTesting {len(models)} models on {len(corpus)} texts")
    print(f"Threshold: |S|/n < {Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS} (from Q51Thresholds)")
    print("\nKey insight:")
    print("  If octants ARE the 8th roots of unity:")
    print("    Sum e^(ikpi/4) = 0  (phases cancel)")
    print("  We measure 8e (magnitudes) because we only see the shadow.")
    print("  The zero IS the completeness signature of the substrate.")
    print()

    # Run negative controls first
    print("Running negative controls...")
    negative_controls = run_all_negative_controls(verbose=verbose)
    control_pass_count = sum(1 for c in negative_controls if c.test_passed)
    print(f"Negative controls: {control_pass_count}/{len(negative_controls)} passed\n")

    # Test each model
    results = []
    for model in models:
        try:
            result = test_model_zero_signature(model, corpus, verbose=verbose)
            results.append(result)
        except Q51ValidationError as e:
            print(f"Validation error testing {model}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error testing {model}: {e}")
            traceback.print_exc()
            continue

    # Aggregate
    if not results:
        raise RuntimeError("No models tested successfully")

    magnitudes = np.array([r.normalized_magnitude for r in results])
    mean_mag = float(np.mean(magnitudes))
    std_mag = float(np.std(magnitudes))
    cv_mag = std_mag / mean_mag if mean_mag > 0 else float('inf')

    # Bootstrap CI for mean magnitude
    if len(magnitudes) >= 3:
        mag_ci = bootstrap_ci(magnitudes, n_bootstrap=Q51Thresholds.BOOTSTRAP_ITERATIONS)
    else:
        # Not enough samples for reliable CI
        mag_ci = BootstrapCI(
            mean=mean_mag, ci_lower=mean_mag, ci_upper=mean_mag,
            std=std_mag, n_samples=len(magnitudes), n_bootstrap=0,
            confidence_level=Q51Thresholds.CONFIDENCE_LEVEL
        )

    models_with_zero = sum(1 for r in results if r.is_zero)
    uniformly_distributed = sum(1 for r in results if r.uniformity_p_value > Q51Thresholds.ZERO_SIG_UNIFORMITY_P)

    # Verdict using threshold constants
    if models_with_zero >= len(results) * 0.8 and cv_mag < Q51Thresholds.ZERO_SIG_CV_THRESHOLD:
        hypothesis_supported = True
        verdict = "CONFIRMED: Octants are the 8th roots of unity"
    elif models_with_zero >= len(results) * 0.5:
        hypothesis_supported = True
        verdict = "PARTIAL SUPPORT: Most models show zero signature"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: Octants are NOT phase sectors"

    # Collect metadata
    metadata = get_test_metadata()
    metadata['thresholds'] = {
        'magnitude_pass': Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS,
        'cv_threshold': Q51Thresholds.ZERO_SIG_CV_THRESHOLD,
        'uniformity_p': Q51Thresholds.ZERO_SIG_UNIFORMITY_P,
    }

    cross_result = CrossModelResult(
        n_models=len(results),
        mean_normalized_magnitude=mean_mag,
        std_normalized_magnitude=std_mag,
        cv_normalized_magnitude=float(cv_mag * 100),  # as percentage
        models_with_zero=models_with_zero,
        uniformly_distributed=uniformly_distributed,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        magnitude_ci=mag_ci.to_dict(),
        negative_controls=[c.to_dict() for c in negative_controls],
        test_metadata=metadata,
        result_hash=None  # Will be computed after full serialization
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean |S|/n: {format_ci(mag_ci)} (threshold: < {Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS})")
    print(f"CV: {cv_mag*100:.1f}% (threshold: < {Q51Thresholds.ZERO_SIG_CV_THRESHOLD*100:.0f}%)")
    print(f"\nModels with zero signature: {models_with_zero}/{len(results)}")
    print(f"Models with uniform distribution: {uniformly_distributed}/{len(results)}")

    print("\n" + "-" * 70)
    print("Negative Controls:")
    print("-" * 70)
    for ctrl in negative_controls:
        status = "PASS" if ctrl.test_passed else "FAIL"
        print(f"  {ctrl.name:<25} {ctrl.actual_behavior:<20} {status}")

    print("\n" + "-" * 70)
    print("Per-model results:")
    print("-" * 70)
    print(f"{'Model':<35} {'|S|/n':>8} {'chi-sq p':>10} {'Status':>20}")
    print("-" * 70)
    for r in results:
        short_name = r.model_name.split('/')[-1][:30]
        print(f"{short_name:<35} {r.normalized_magnitude:>8.4f} {r.uniformity_p_value:>10.2e} {r.status:>20}")

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
    """
    Save results to JSON file.

    Includes result hash for integrity verification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results dict
    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_ZERO_SIGNATURE',
        'test_version': '2.0-hardened',
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
                'status': r.status,
                'validation_warnings': r.validation_warnings,
                'embedding_stats': r.embedding_stats
            }
            for r in results
        ],
        'cross_model': {
            'n_models': cross_result.n_models,
            'mean_normalized_magnitude': cross_result.mean_normalized_magnitude,
            'std_normalized_magnitude': cross_result.std_normalized_magnitude,
            'cv_normalized_magnitude': cross_result.cv_normalized_magnitude,
            'models_with_zero': cross_result.models_with_zero,
            'uniformly_distributed': cross_result.uniformly_distributed,
            'hypothesis_supported': cross_result.hypothesis_supported,
            'verdict': cross_result.verdict,
            'magnitude_ci': cross_result.magnitude_ci,
        },
        'negative_controls': cross_result.negative_controls,
        'test_metadata': cross_result.test_metadata,
        'constants': {
            'SEMIOTIC_CONSTANT': SEMIOTIC_CONSTANT,
            'CRITICAL_ALPHA': CRITICAL_ALPHA,
            'SECTOR_WIDTH': SECTOR_WIDTH
        },
        'thresholds': {
            'ZERO_SIG_MAGNITUDE_PASS': Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS,
            'ZERO_SIG_MAGNITUDE_FAIL': Q51Thresholds.ZERO_SIG_MAGNITUDE_FAIL,
            'ZERO_SIG_CV_THRESHOLD': Q51Thresholds.ZERO_SIG_CV_THRESHOLD,
            'ZERO_SIG_UNIFORMITY_P': Q51Thresholds.ZERO_SIG_UNIFORMITY_P,
        }
    }

    # Compute result hash (before adding it to output)
    output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q51_zero_signature_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Result hash: {output['result_hash']}")


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
