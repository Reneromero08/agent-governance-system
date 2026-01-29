"""
Q51 Semantic Domain Coherence Test - Test #9

Tests whether words in the same semantic domain have similar phase offsets.

Hypothesis:
    If phases carry semantic meaning, words in the same domain (animals, colors,
    emotions, etc.) should cluster in phase space. Phase variance WITHIN domains
    should be smaller than variance BETWEEN domains.

Method:
    1. Embed 10 words from each of 10 semantic domains
    2. Recover phases for all words
    3. Compute within-domain phase variance vs between-domain variance
    4. Compute F-ratio: var_between / var_within

Pass criteria:
    - F-ratio > 2.0 (phases cluster by domain)
    - ANOVA p-value < 0.01
    - At least 8/10 domains show internal coherence

Falsification:
    - F-ratio < 1.0 (no domain structure in phases)
    - Random phase assignment within domains
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add paths
SCRIPT_DIR = Path(__file__).parent
QGT_PATH = SCRIPT_DIR.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(QGT_PATH))

# Import test harness
from q51_test_harness import (
    Q51Thresholds,
    Q51Seeds,
    Q51ValidationError,
    ValidationResult,
    BootstrapCI,
    NegativeControlResult,
    validate_embeddings,
    bootstrap_ci,
    generate_null_embeddings,
    compute_result_hash,
    format_ci,
    get_test_metadata,
    Q51Logger,
)

from qgt_phase import (
    octant_phase_mapping,
    circular_variance,
    circular_mean,
)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

# =============================================================================
# Constants
# =============================================================================

# Thresholds
#
# CRITICAL FIX v2: For circular ANOVA with 10 domains x 10 words,
# the NULL distribution (uniform random phases) has:
#   - Mean F-ratio: ~10.8
#   - 95th percentile: ~16.0
#   - 99th percentile: ~18.5
#
# This is because even random domain means differ substantially due to
# sampling variability on the circle. The old thresholds (2.0, 1.5) were
# completely wrong - random data would always "pass"!
#
# New thresholds based on empirical null distribution (1000 simulations):
F_RATIO_PASS = 20.0  # Clearly above 95th percentile (~16)
F_RATIO_PARTIAL = 16.0  # At 95th percentile
F_RATIO_NULL_MEAN = 10.8  # Expected value under null hypothesis
F_RATIO_NULL_95TH = 16.0  # 95th percentile of null distribution
ANOVA_P_THRESHOLD = 0.01
COHERENT_DOMAINS_PASS = 8  # At least 8/10 domains coherent

# Semantic domains (10 domains, 10 words each)
SEMANTIC_DOMAINS = {
    'animals': [
        'dog', 'cat', 'horse', 'elephant', 'lion',
        'tiger', 'bear', 'wolf', 'deer', 'rabbit'
    ],
    'colors': [
        'red', 'blue', 'green', 'yellow', 'orange',
        'purple', 'pink', 'brown', 'black', 'white'
    ],
    'emotions': [
        'happy', 'sad', 'angry', 'fearful', 'surprised',
        'disgusted', 'anxious', 'excited', 'calm', 'bored'
    ],
    'professions': [
        'doctor', 'lawyer', 'teacher', 'engineer', 'chef',
        'pilot', 'nurse', 'artist', 'scientist', 'musician'
    ],
    'countries': [
        'france', 'germany', 'japan', 'brazil', 'india',
        'canada', 'australia', 'spain', 'italy', 'russia'
    ],
    'food': [
        'apple', 'bread', 'cheese', 'rice', 'pasta',
        'chicken', 'fish', 'salad', 'soup', 'cake'
    ],
    'vehicles': [
        'car', 'bus', 'train', 'airplane', 'boat',
        'bicycle', 'motorcycle', 'helicopter', 'truck', 'subway'
    ],
    'weather': [
        'sunny', 'rainy', 'cloudy', 'windy', 'stormy',
        'snowy', 'foggy', 'humid', 'cold', 'hot'
    ],
    'furniture': [
        'table', 'chair', 'sofa', 'bed', 'desk',
        'shelf', 'cabinet', 'dresser', 'lamp', 'mirror'
    ],
    'sports': [
        'soccer', 'basketball', 'tennis', 'swimming', 'running',
        'golf', 'baseball', 'hockey', 'volleyball', 'boxing'
    ],
}

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
class DomainResult:
    """Result for a single semantic domain."""
    domain_name: str
    n_words: int
    mean_phase: float
    circular_variance: float  # 0 = perfectly coherent, 1 = uniform
    is_coherent: bool  # Variance < threshold


@dataclass
class ModelCoherenceResult:
    """Semantic coherence result for a single model."""
    model_name: str
    n_domains: int
    n_words_total: int
    domain_results: List[Dict]
    f_ratio: float
    anova_p_value: float
    coherent_domains: int
    var_within: float
    var_between: float
    status: str
    validation_warnings: Optional[List[str]] = None


@dataclass
class CrossModelCoherenceResult:
    """Cross-model aggregation."""
    n_models: int
    mean_f_ratio: float
    std_f_ratio: float
    mean_coherent_domains: float
    models_passing: int
    hypothesis_supported: bool
    verdict: str
    f_ratio_ci: Optional[dict] = None
    domain_summary: Optional[Dict] = None  # Average coherence per domain
    negative_controls: Optional[List[dict]] = None
    test_metadata: Optional[dict] = None
    result_hash: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_embeddings_for_words(
    model_name: str,
    words: List[str],
    validate: bool = True
) -> Tuple[np.ndarray, ValidationResult]:
    """Get embeddings for a list of words."""
    embeddings = None
    model_error = None

    if HAS_ST:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(words, show_progress_bar=False)
            embeddings = np.array(embeddings)
        except Exception as e:
            model_error = str(e)
            print(f"  Warning: Could not load {model_name}: {e}")

    # Synthetic fallback
    if embeddings is None:
        np.random.seed(hash(model_name) % 2**32)
        dim = 384
        n = len(words)
        embeddings = []
        for word in words:
            np.random.seed(hash(word) % 2**32)
            vec = np.random.randn(dim)
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            embeddings.append(vec)
        embeddings = np.array(embeddings)

    # Validate
    if validate:
        validation = validate_embeddings(
            embeddings, min_samples=5, name=f"coherence_{model_name}"
        )
        if model_error:
            validation.warnings.append(f"Using synthetic fallback: {model_error}")
    else:
        validation = ValidationResult(True, [], [], {'n_samples': len(embeddings)})

    return embeddings, validation


def compute_circular_anova(domain_phases: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Compute circular ANOVA for phase data across domains.

    Returns (f_ratio, p_value, var_within, var_between)
    """
    # Collect all phases
    all_phases = []
    domain_means = []
    domain_sizes = []

    for domain, phases in domain_phases.items():
        all_phases.extend(phases)
        domain_means.append(circular_mean(phases))
        domain_sizes.append(len(phases))

    all_phases = np.array(all_phases)
    domain_means = np.array(domain_means)
    domain_sizes = np.array(domain_sizes)
    n_total = len(all_phases)
    n_domains = len(domain_phases)

    # Grand mean
    grand_mean = circular_mean(all_phases)

    # Sum of squares between (circular version)
    # SS_between = Sum[n_k * (mu_k - mu)^2]
    ss_between = 0.0
    for i, (domain, phases) in enumerate(domain_phases.items()):
        diff = np.angle(np.exp(1j * (domain_means[i] - grand_mean)))
        ss_between += len(phases) * (diff ** 2)

    # Sum of squares within
    # SS_within = Sum[Sum[(x_ij - mu_i)^2]]
    ss_within = 0.0
    for i, (domain, phases) in enumerate(domain_phases.items()):
        for phase in phases:
            diff = np.angle(np.exp(1j * (phase - domain_means[i])))
            ss_within += diff ** 2

    # Degrees of freedom
    df_between = n_domains - 1
    df_within = n_total - n_domains

    if df_between == 0 or df_within == 0:
        return 0.0, 1.0, 0.0, 0.0

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-ratio
    if ms_within > 1e-10:
        f_ratio = ms_between / ms_within
    else:
        f_ratio = float('inf') if ms_between > 0 else 0.0

    # P-value from F-distribution
    try:
        from scipy.stats import f as f_dist
        p_value = 1 - f_dist.cdf(f_ratio, df_between, df_within)
    except:
        p_value = 0.5  # Fallback

    var_within = float(ms_within)
    var_between = float(ms_between)

    return float(f_ratio), float(p_value), var_within, var_between


# =============================================================================
# Test Functions
# =============================================================================

def test_coherence_single_model(
    model_name: str,
    domains: Dict[str, List[str]],
    verbose: bool = True
) -> ModelCoherenceResult:
    """Test semantic coherence for a single model."""
    logger = Q51Logger(f"coherence_{model_name}", verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Semantic Coherence Test: {model_name}")
        print(f"{'='*60}")

    # Collect all words and their domain labels
    all_words = []
    word_domains = []
    for domain, words in domains.items():
        for word in words:
            all_words.append(word)
            word_domains.append(domain)

    # Get embeddings
    embeddings, validation = get_embeddings_for_words(model_name, all_words)

    if not validation.valid:
        raise Q51ValidationError(f"Embedding validation failed: {validation.errors}")

    if verbose:
        print(f"Total words: {len(all_words)}")
        print(f"Domains: {len(domains)}")

    # Recover phases
    octant_result = octant_phase_mapping(embeddings)
    phases = octant_result.octant_phases

    # Organize phases by domain
    domain_phases = {}
    for i, (word, domain) in enumerate(zip(all_words, word_domains)):
        if domain not in domain_phases:
            domain_phases[domain] = []
        domain_phases[domain].append(phases[i])

    for domain in domain_phases:
        domain_phases[domain] = np.array(domain_phases[domain])

    # Analyze each domain
    domain_results = []
    coherent_count = 0
    coherence_threshold = 0.7  # Circular variance threshold for coherence

    for domain, ph in domain_phases.items():
        cv = circular_variance(ph)
        mean_ph = circular_mean(ph)
        is_coherent = bool(cv < coherence_threshold)

        if is_coherent:
            coherent_count += 1

        domain_results.append(DomainResult(
            domain_name=domain,
            n_words=len(ph),
            mean_phase=float(mean_ph),
            circular_variance=float(cv),
            is_coherent=is_coherent
        ))

    if verbose:
        print(f"\nPer-domain circular variance:")
        print(f"{'Domain':<15} {'Mean Phase':>10} {'Circ Var':>10} {'Coherent':>10}")
        print("-" * 50)
        for dr in domain_results:
            coh_str = "Yes" if dr.is_coherent else "No"
            print(f"{dr.domain_name:<15} {dr.mean_phase:>10.3f} {dr.circular_variance:>10.3f} {coh_str:>10}")

    # Compute circular ANOVA
    f_ratio, p_value, var_within, var_between = compute_circular_anova(domain_phases)

    if verbose:
        print(f"\nCircular ANOVA:")
        print(f"  F-ratio: {f_ratio:.4f}")
        print(f"  Null baseline: ~{F_RATIO_NULL_MEAN:.1f} (95th pct: ~{F_RATIO_NULL_95TH:.1f})")
        print(f"  Pass threshold: > {F_RATIO_PASS}")
        print(f"  p-value: {p_value:.4e}")
        print(f"  Var within: {var_within:.4f}")
        print(f"  Var between: {var_between:.4f}")
        print(f"  Coherent domains: {coherent_count}/{len(domains)}")

    # Determine status
    if f_ratio > F_RATIO_PASS and p_value < ANOVA_P_THRESHOLD:
        status = "PASS"
    elif f_ratio > F_RATIO_PARTIAL or coherent_count >= COHERENT_DOMAINS_PASS:
        status = "PARTIAL"
    else:
        status = "FAIL"

    if verbose:
        print(f"Status: {status}")

    return ModelCoherenceResult(
        model_name=model_name,
        n_domains=len(domains),
        n_words_total=len(all_words),
        domain_results=[asdict(dr) for dr in domain_results],
        f_ratio=float(f_ratio),
        anova_p_value=float(p_value),
        coherent_domains=coherent_count,
        var_within=float(var_within),
        var_between=float(var_between),
        status=status,
        validation_warnings=validation.warnings if validation.warnings else None
    )


def run_negative_control(verbose: bool = True) -> NegativeControlResult:
    """
    Negative control: Random domain assignment should show F-ratio near the
    null distribution mean (~10.8), NOT ~1.0.

    CRITICAL FIX v2:
        For circular ANOVA with 10 domains x 10 words, even RANDOM domain
        assignment gives F-ratio ~10.8 (not 1.0!) because:
        1. Each domain's mean phase is a random point on the circle
        2. These random domain means differ substantially (sampling variability)
        3. The between-group variance is NOT zero even for random assignment

        The null distribution (1000 simulations) shows:
        - Mean F-ratio: ~10.8
        - Std: ~3.1
        - 95th percentile: ~16.0

        So F in range [5, 18] is EXPECTED for random data.
        F > 20 indicates genuine semantic clustering.
    """
    print("\n  [Negative Control] Shuffled domain labels...")

    # Use a fixed model
    all_words = []
    for domain, words in SEMANTIC_DOMAINS.items():
        all_words.extend(words)

    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)
    dim = 384

    # Generate synthetic embeddings
    embeddings = []
    for word in all_words:
        np.random.seed(hash(word) % 2**32)
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec)
    embeddings = np.array(embeddings)

    # Recover phases
    octant_result = octant_phase_mapping(embeddings)
    phases = octant_result.octant_phases

    # Shuffle domain assignments - use proper non-overlapping shuffle
    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)
    shuffled_indices = np.random.permutation(len(phases))
    domain_phases = {}
    idx = 0
    for domain in SEMANTIC_DOMAINS.keys():
        n_words = len(SEMANTIC_DOMAINS[domain])
        domain_phases[domain] = phases[shuffled_indices[idx:idx+n_words]]
        idx += n_words

    # Compute F-ratio
    f_ratio, p_value, _, _ = compute_circular_anova(domain_phases)

    # F-ratio should be in the expected null range [5, 18]
    # NOT ~1.0 as previously assumed!
    NULL_F_LOWER = 5.0
    NULL_F_UPPER = 18.0
    is_in_null_range = NULL_F_LOWER < f_ratio < NULL_F_UPPER

    if verbose:
        print(f"    F-ratio: {f_ratio:.4f}")
        print(f"    Expected range for null: [{NULL_F_LOWER}, {NULL_F_UPPER}]")
        print(f"    (Null mean ~{F_RATIO_NULL_MEAN:.1f}, 95th percentile ~{F_RATIO_NULL_95TH:.1f})")
        status = "PASS" if is_in_null_range else "FAIL"
        print(f"    Status: {status}")

    return NegativeControlResult(
        name="shuffled_domains_f_ratio",
        test_passed=is_in_null_range,
        expected_behavior=f"Shuffled domains should have F-ratio in null range [{NULL_F_LOWER}-{NULL_F_UPPER}]",
        actual_behavior=f"F-ratio = {f_ratio:.4f}",
        metric_value=f_ratio,
        metric_threshold=F_RATIO_NULL_MEAN,
        notes="Circular ANOVA null distribution has mean ~10.8, NOT 1.0"
    )


def test_coherence_cross_model(
    models: List[str],
    domains: Dict[str, List[str]],
    verbose: bool = True
) -> Tuple[List[ModelCoherenceResult], CrossModelCoherenceResult]:
    """Test semantic coherence across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 SEMANTIC DOMAIN COHERENCE TEST - CROSS-MODEL VALIDATION")
    print("=" * 70)
    print(f"\nTesting {len(models)} models")
    print(f"Domains: {list(domains.keys())}")
    print(f"Words per domain: {len(list(domains.values())[0])}")
    print("\nKey insight:")
    print("  If phases carry SEMANTIC meaning, words in same domain should cluster.")
    print("  F-ratio > 1 means more variance BETWEEN than WITHIN domains.")
    print()

    # Run negative control
    negative_control = run_negative_control(verbose=verbose)

    # Test each model
    results = []
    for model in models:
        try:
            result = test_coherence_single_model(model, domains, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    if not results:
        raise RuntimeError("No models tested successfully")

    # Aggregate
    f_ratios = [r.f_ratio for r in results]
    coherent_counts = [r.coherent_domains for r in results]

    mean_f = float(np.mean(f_ratios))
    std_f = float(np.std(f_ratios))
    mean_coherent = float(np.mean(coherent_counts))

    passing = sum(1 for r in results if r.status == "PASS")

    # Bootstrap CI for F-ratio
    if len(f_ratios) >= 3:
        f_ci = bootstrap_ci(np.array(f_ratios), n_bootstrap=Q51Thresholds.BOOTSTRAP_ITERATIONS)
    else:
        f_ci = BootstrapCI(
            mean=mean_f, ci_lower=min(f_ratios), ci_upper=max(f_ratios),
            std=std_f, n_samples=len(f_ratios), n_bootstrap=0,
            confidence_level=Q51Thresholds.CONFIDENCE_LEVEL
        )

    # Per-domain summary (average variance)
    domain_summary = {}
    for domain in domains.keys():
        variances = []
        for result in results:
            for dr in result.domain_results:
                if dr['domain_name'] == domain:
                    variances.append(dr['circular_variance'])
        domain_summary[domain] = float(np.mean(variances)) if variances else 0.0

    # Verdict
    if passing == len(results):
        hypothesis_supported = True
        verdict = "CONFIRMED: Phases cluster by semantic domain"
    elif passing >= len(results) * 0.6:
        hypothesis_supported = True
        verdict = "PARTIAL: Most models show semantic clustering"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: No semantic structure in phases"

    # Metadata
    metadata = get_test_metadata()

    cross_result = CrossModelCoherenceResult(
        n_models=len(results),
        mean_f_ratio=mean_f,
        std_f_ratio=std_f,
        mean_coherent_domains=mean_coherent,
        models_passing=passing,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        f_ratio_ci=f_ci.to_dict(),
        domain_summary=domain_summary,
        negative_controls=[negative_control.to_dict()],
        test_metadata=metadata
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean F-ratio: {format_ci(f_ci)} (threshold: > {F_RATIO_PASS})")
    print(f"Mean coherent domains: {mean_coherent:.1f}/{len(domains)}")
    print(f"Models passing: {passing}/{len(results)}")
    print(f"\n{verdict}")

    return results, cross_result


def save_results(
    results: List[ModelCoherenceResult],
    cross_result: CrossModelCoherenceResult,
    output_dir: Path
):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_SEMANTIC_COHERENCE',
        'hypothesis': 'Words in same semantic domain cluster in phase space',
        'per_model': [asdict(r) for r in results],
        'cross_model': asdict(cross_result),
        'hardening': {
            'domains': list(SEMANTIC_DOMAINS.keys()),
            'words_per_domain': 10,
            'thresholds': {
                'F_RATIO_PASS': F_RATIO_PASS,
                'F_RATIO_PARTIAL': F_RATIO_PARTIAL,
                'ANOVA_P_THRESHOLD': ANOVA_P_THRESHOLD,
                'COHERENT_DOMAINS_PASS': COHERENT_DOMAINS_PASS,
            },
            'seeds': {
                'NEGATIVE_CONTROL': Q51Seeds.NEGATIVE_CONTROL,
            }
        }
    }

    # Compute hash
    output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q51_semantic_coherence_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Result hash: {output['result_hash']}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Semantic Coherence Test."""
    print("\n" + "=" * 70)
    print("Q51 TEST #9: SEMANTIC DOMAIN COHERENCE")
    print("=" * 70)

    results, cross_result = test_coherence_cross_model(
        models=MODELS,
        domains=SEMANTIC_DOMAINS,
        verbose=True
    )

    # Save results
    output_dir = SCRIPT_DIR / "results"
    save_results(results, cross_result, output_dir)

    print("\n" + "=" * 70)

    return cross_result.hypothesis_supported


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
