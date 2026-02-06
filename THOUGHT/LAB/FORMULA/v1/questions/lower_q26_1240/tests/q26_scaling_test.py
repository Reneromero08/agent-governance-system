#!/usr/bin/env python3
"""
Q26: RIGOROUS Multi-Model Scaling Test for Minimum Data Requirements

PROBLEM WITH ORIGINAL TEST:
- Hypothesis failed (no scaling law found, all R^2 < 0.5)
- But document pivots to "N=5-10 is enough" based on SINGLE test at D=384
- This is SPIN - underpowered and misleading

THIS TEST:
1. Tests MULTIPLE real embedding models with different dimensionalities
2. Uses proper statistical methodology
3. Reports the ACTUAL scaling law (or lack thereof) HONESTLY
4. If there's no universal N_min, we say so

Models to test:
- all-MiniLM-L6-v2 (D=384)
- all-mpnet-base-v2 (D=768)
- paraphrase-MiniLM-L3-v2 (D=384, different architecture)
- all-distilroberta-v1 (D=768)
- PCA projections of 768-dim to 50, 100, 200 dimensions

Author: Claude (rigorous retest)
Date: 2026-01-27
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("ERROR: SentenceTransformer required for real embedding tests")


@dataclass
class StabilityResult:
    """Result of stability test at specific N and model."""
    model_name: str
    D: int
    N: int
    R_mean: float
    R_std: float
    R_cv: float
    n_trials: int
    is_stable: bool
    R_values: List[float]


@dataclass
class ModelScalingResult:
    """Scaling analysis for a specific model."""
    model_name: str
    D: int
    N_min: int  # Smallest N where CV < threshold
    stability_curve: Dict[int, float]  # N -> CV mapping
    all_results: List[StabilityResult]


def compute_R(embeddings: np.ndarray) -> float:
    """
    Compute R from embeddings using centroid-based method.

    R = E * concentration / sigma
    Where:
      - truth = centroid of embeddings
      - distances = ||embeddings - truth||
      - sigma = mean(distances)
      - z = distances / sigma (normalized, ~1.0)
      - E = mean(exp(-0.5 * z^2)) (Gaussian evidence)
      - cv = std(distances) / mean(distances)
      - concentration = 1 / (1 + cv)
    """
    if len(embeddings) < 2:
        return 1.0

    # Compute centroid (truth vector)
    truth_vector = embeddings.mean(axis=0)

    # Compute distances from centroid
    distances = np.linalg.norm(embeddings - truth_vector, axis=1)
    mean_dist = np.mean(distances)

    if mean_dist < 1e-10:
        return float(len(embeddings))

    # Scale parameter
    sigma = mean_dist

    # Normalized distances (z ~ 1.0 by construction)
    z = distances / sigma

    # Gaussian evidence
    E = np.mean(np.exp(-0.5 * z**2))

    # Concentration factor
    cv = np.std(distances) / (mean_dist + 1e-10)
    concentration = 1.0 / (1.0 + cv)

    # R formula
    R = float(E * concentration / sigma)

    return max(0.0, min(R, 1000.0))


def project_to_dimension(embeddings: np.ndarray, target_dim: int, seed: int = 42) -> np.ndarray:
    """Project embeddings to target dimension using PCA."""
    if embeddings.shape[1] <= target_dim:
        return embeddings

    # Center the data
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance and get eigenvectors
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project to top dimensions
    projection_matrix = eigenvectors[:, :target_dim]
    projected = centered @ projection_matrix

    return projected


def test_stability(
    embeddings: np.ndarray,
    N: int,
    n_trials: int = 50,
    stability_threshold: float = 0.10
) -> StabilityResult:
    """
    Test R stability at sample size N with bootstrap resampling.

    Args:
        embeddings: Full embedding matrix (n_samples x D)
        N: Sample size to test
        n_trials: Number of bootstrap trials
        stability_threshold: CV threshold for stability (default 10%)

    Returns:
        StabilityResult with CV and stability assessment
    """
    if N > len(embeddings):
        N = len(embeddings)

    R_values = []
    for trial in range(n_trials):
        np.random.seed(trial * 1000 + N)
        idx = np.random.choice(len(embeddings), N, replace=False)
        R = compute_R(embeddings[idx])
        R_values.append(R)

    R_array = np.array(R_values)
    R_mean = float(np.mean(R_array))
    R_std = float(np.std(R_array))
    R_cv = R_std / (R_mean + 1e-10)

    return StabilityResult(
        model_name="",  # To be filled by caller
        D=embeddings.shape[1],
        N=N,
        R_mean=R_mean,
        R_std=R_std,
        R_cv=R_cv,
        n_trials=n_trials,
        is_stable=R_cv < stability_threshold,
        R_values=R_values
    )


def find_N_min(
    embeddings: np.ndarray,
    N_candidates: List[int],
    n_trials: int = 50,
    stability_threshold: float = 0.10
) -> Tuple[int, Dict[int, float]]:
    """
    Find minimum N for stable R.

    Returns:
        Tuple of (N_min, stability_curve)
    """
    stability_curve = {}

    for N in N_candidates:
        if N > len(embeddings):
            break
        result = test_stability(embeddings, N, n_trials, stability_threshold)
        stability_curve[N] = result.R_cv
        if result.is_stable and result.N not in stability_curve:
            pass  # Continue to build full curve

    # Find first stable N
    N_min = None
    for N in sorted(stability_curve.keys()):
        if stability_curve[N] < stability_threshold:
            N_min = N
            break

    if N_min is None:
        N_min = max(stability_curve.keys()) if stability_curve else -1

    return N_min, stability_curve


def generate_semantic_corpus(n_texts: int = 200) -> List[str]:
    """Generate a diverse corpus for embedding tests."""
    # Semantic clusters to ensure structure in the embeddings
    topics = [
        # Science
        [
            "The laws of physics govern the universe.",
            "Quantum mechanics describes atomic behavior.",
            "Einstein's relativity changed our view of time.",
            "Energy and matter are interchangeable.",
            "The speed of light is a universal constant.",
            "Entropy always increases in closed systems.",
            "Gravity bends the fabric of spacetime.",
            "Particles exhibit wave-like behavior.",
        ],
        # Nature
        [
            "Trees convert carbon dioxide to oxygen.",
            "Rivers flow from mountains to the sea.",
            "Forests provide habitat for wildlife.",
            "The ocean covers most of Earth's surface.",
            "Seasons change as Earth orbits the sun.",
            "Ecosystems maintain delicate balance.",
            "Plants grow toward sunlight.",
            "Animals adapt to their environments.",
        ],
        # Technology
        [
            "Computers process information rapidly.",
            "The internet connects billions of people.",
            "Artificial intelligence learns from data.",
            "Software automates repetitive tasks.",
            "Networks enable global communication.",
            "Algorithms solve complex problems.",
            "Data storage continues to improve.",
            "Machine learning finds hidden patterns.",
        ],
        # Philosophy
        [
            "Knowledge requires justified true belief.",
            "Ethics guides moral decision making.",
            "Logic distinguishes valid from invalid reasoning.",
            "Consciousness remains deeply mysterious.",
            "Free will is debated by philosophers.",
            "Truth and reality are fundamental concepts.",
            "Meaning emerges from human experience.",
            "Wisdom comes from reflection and study.",
        ],
        # Daily life
        [
            "Breakfast provides morning energy.",
            "Exercise improves physical health.",
            "Sleep restores body and mind.",
            "Cooking is both art and science.",
            "Music evokes powerful emotions.",
            "Reading expands mental horizons.",
            "Friends share joys and sorrows.",
            "Work gives purpose and income.",
        ],
        # History
        [
            "Ancient civilizations built great monuments.",
            "Wars reshape political boundaries.",
            "Inventions change the course of history.",
            "Leaders influence their times.",
            "Cultures preserve traditions over generations.",
            "Trade routes connected distant peoples.",
            "Revolutions transform societies.",
            "History teaches lessons about humanity.",
        ],
        # Mathematics
        [
            "Numbers are the foundation of mathematics.",
            "Geometry describes spatial relationships.",
            "Algebra uses symbols for unknown quantities.",
            "Calculus measures rates of change.",
            "Statistics reveals patterns in data.",
            "Proofs establish mathematical truth.",
            "Infinity stretches beyond counting.",
            "Equations express precise relationships.",
        ],
        # Art
        [
            "Painting captures moments in color.",
            "Sculpture gives form to imagination.",
            "Poetry distills language to essence.",
            "Music expresses emotion without words.",
            "Dance tells stories through movement.",
            "Architecture shapes the spaces we inhabit.",
            "Film combines many artistic forms.",
            "Theater brings stories to life.",
        ],
    ]

    # Flatten and extend
    all_texts = []
    for topic in topics:
        all_texts.extend(topic)

    # Generate variations
    prefixes = ["Indeed, ", "Actually, ", "In fact, ", "Clearly, ", "Obviously, "]
    suffixes = [" This is important.", " This matters.", " This is true.", " This is fundamental."]

    extended_texts = list(all_texts)
    for text in all_texts:
        for prefix in prefixes[:2]:
            extended_texts.append(prefix + text.lower())
        for suffix in suffixes[:2]:
            extended_texts.append(text.rstrip('.') + suffix)

    # Ensure we have enough texts
    while len(extended_texts) < n_texts:
        extended_texts.extend(all_texts)

    return extended_texts[:n_texts]


def run_model_test(
    model_name: str,
    texts: List[str],
    N_candidates: List[int],
    target_dim: Optional[int] = None,
    n_trials: int = 50
) -> ModelScalingResult:
    """
    Run scaling test for a specific model.

    Args:
        model_name: Name of SentenceTransformer model
        texts: List of texts to embed
        N_candidates: Sample sizes to test
        target_dim: If set, project to this dimension
        n_trials: Bootstrap trials per N

    Returns:
        ModelScalingResult with N_min and stability curve
    """
    print(f"\n  Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"  Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=False)

    original_dim = embeddings.shape[1]

    # Project if needed
    if target_dim and target_dim < original_dim:
        print(f"  Projecting {original_dim}D -> {target_dim}D via PCA...")
        embeddings = project_to_dimension(embeddings, target_dim)
        D = target_dim
        display_name = f"{model_name}_PCA{target_dim}"
    else:
        D = original_dim
        display_name = model_name

    print(f"  Testing stability at D={D}...")

    # Filter N_candidates to valid range
    valid_N = [n for n in N_candidates if n <= len(texts)]

    N_min, stability_curve = find_N_min(embeddings, valid_N, n_trials)

    # Collect all results
    all_results = []
    for N in valid_N:
        result = test_stability(embeddings, N, n_trials)
        result.model_name = display_name
        all_results.append(result)

    return ModelScalingResult(
        model_name=display_name,
        D=D,
        N_min=N_min,
        stability_curve=stability_curve,
        all_results=all_results
    )


def fit_scaling_law(
    D_values: np.ndarray,
    N_min_values: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Fit multiple scaling laws and return R^2 for each.

    Tests:
    - Log: N_min = c * log(D) + b
    - Linear: N_min = c * D + b
    - Sqrt: N_min = c * sqrt(D) + b
    - Constant: N_min = c (no scaling)
    """
    results = {}

    # Log scaling
    log_D = np.log(D_values)
    try:
        c, b = np.polyfit(log_D, N_min_values, 1)
        predicted = c * log_D + b
        ss_res = np.sum((N_min_values - predicted) ** 2)
        ss_tot = np.sum((N_min_values - np.mean(N_min_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        results['log'] = {'c': float(c), 'b': float(b), 'r2': float(r2)}
    except Exception:
        results['log'] = {'c': 0, 'b': 0, 'r2': 0}

    # Linear scaling
    try:
        c, b = np.polyfit(D_values, N_min_values, 1)
        predicted = c * D_values + b
        ss_res = np.sum((N_min_values - predicted) ** 2)
        ss_tot = np.sum((N_min_values - np.mean(N_min_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        results['linear'] = {'c': float(c), 'b': float(b), 'r2': float(r2)}
    except Exception:
        results['linear'] = {'c': 0, 'b': 0, 'r2': 0}

    # Sqrt scaling
    sqrt_D = np.sqrt(D_values)
    try:
        c, b = np.polyfit(sqrt_D, N_min_values, 1)
        predicted = c * sqrt_D + b
        ss_res = np.sum((N_min_values - predicted) ** 2)
        ss_tot = np.sum((N_min_values - np.mean(N_min_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        results['sqrt'] = {'c': float(c), 'b': float(b), 'r2': float(r2)}
    except Exception:
        results['sqrt'] = {'c': 0, 'b': 0, 'r2': 0}

    # Constant (no scaling) - just compute variance
    mean_N = np.mean(N_min_values)
    std_N = np.std(N_min_values)
    cv = std_N / (mean_N + 1e-10)
    # R^2 is 0 for constant model by definition (explains nothing)
    results['constant'] = {'c': float(mean_N), 'b': 0, 'r2': 0, 'cv': float(cv)}

    return results


def run_comprehensive_test():
    """
    Run comprehensive multi-model scaling test.
    """
    print("=" * 80)
    print("Q26: RIGOROUS MULTI-MODEL SCALING TEST")
    print("=" * 80)
    print("\nObjective: Test if there is a universal N_min scaling law")
    print("Methodology: Multiple real models, multiple dimensionalities")

    if not ST_AVAILABLE:
        print("\nERROR: SentenceTransformer not available. Cannot run test.")
        return None

    # Generate corpus
    print("\n" + "-" * 60)
    print("PHASE 1: GENERATING TEST CORPUS")
    print("-" * 60)
    texts = generate_semantic_corpus(200)
    print(f"Generated {len(texts)} texts with semantic structure")

    # Define models and configurations
    models_config = [
        # Native dimensionalities
        ('all-MiniLM-L6-v2', None),           # D=384
        ('all-mpnet-base-v2', None),           # D=768
        ('paraphrase-MiniLM-L3-v2', None),     # D=384
        # PCA projections from 768D model
        ('all-mpnet-base-v2', 50),
        ('all-mpnet-base-v2', 100),
        ('all-mpnet-base-v2', 200),
        ('all-mpnet-base-v2', 400),
    ]

    N_candidates = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    n_trials = 50

    print("\n" + "-" * 60)
    print("PHASE 2: TESTING MODELS")
    print("-" * 60)

    model_results = []

    for model_name, target_dim in models_config:
        try:
            result = run_model_test(model_name, texts, N_candidates, target_dim, n_trials)
            model_results.append(result)

            # Print summary
            stable_N = [N for N, cv in sorted(result.stability_curve.items()) if cv < 0.10]
            print(f"  Result: D={result.D}, N_min={result.N_min}")
            print(f"    Stability curve: {dict((n, round(cv, 3)) for n, cv in sorted(result.stability_curve.items())[:5])}")
        except Exception as e:
            print(f"  ERROR: {model_name} failed: {e}")

    print("\n" + "-" * 60)
    print("PHASE 3: SCALING LAW ANALYSIS")
    print("-" * 60)

    # Extract D and N_min for scaling analysis
    D_values = np.array([r.D for r in model_results])
    N_min_values = np.array([r.N_min for r in model_results])

    print("\nData points:")
    for r in model_results:
        print(f"  {r.model_name}: D={r.D}, N_min={r.N_min}")

    # Fit scaling laws
    scaling_fits = fit_scaling_law(D_values, N_min_values)

    print("\nScaling Law Fits:")
    for law, params in scaling_fits.items():
        if law == 'constant':
            print(f"  {law.upper()}: N_min = {params['c']:.1f} (CV = {params['cv']:.2%})")
        else:
            print(f"  {law.upper()}: N_min = {params['c']:.4f} * {law}(D) + {params['b']:.2f} (R^2 = {params['r2']:.4f})")

    # Determine best fit
    best_fit = max(scaling_fits.keys(), key=lambda k: scaling_fits[k]['r2'])
    best_r2 = scaling_fits[best_fit]['r2']

    print("\n" + "-" * 60)
    print("PHASE 4: HONEST VERDICT")
    print("-" * 60)

    # Check for model-specific variation
    N_min_cv = np.std(N_min_values) / (np.mean(N_min_values) + 1e-10)
    N_min_range = (int(np.min(N_min_values)), int(np.max(N_min_values)))

    print(f"\nN_min statistics:")
    print(f"  Range: {N_min_range[0]} to {N_min_range[1]}")
    print(f"  Mean: {np.mean(N_min_values):.1f}")
    print(f"  CV: {N_min_cv:.2%}")

    # Determine verdict
    if best_r2 > 0.7:
        verdict = f"SCALING_LAW_FOUND"
        explanation = f"{best_fit.upper()} scaling explains {best_r2:.1%} of variance"
    elif best_r2 > 0.3:
        verdict = "WEAK_SCALING"
        explanation = f"Best fit ({best_fit}) has R^2={best_r2:.3f} - too weak to confirm"
    elif N_min_cv < 0.20:
        verdict = "NO_SCALING_CONSTANT_N_MIN"
        explanation = f"N_min is roughly constant ({np.mean(N_min_values):.0f} +/- {np.std(N_min_values):.0f}), no D dependence"
    else:
        verdict = "NO_UNIVERSAL_SCALING"
        explanation = f"No scaling law (best R^2={best_r2:.3f}), high variation (CV={N_min_cv:.1%})"

    print(f"\nVERDICT: {verdict}")
    print(f"Explanation: {explanation}")

    # Check if the original claim "N=5-10 is enough" holds
    print("\n" + "-" * 60)
    print("PHASE 5: TESTING ORIGINAL CLAIM")
    print("-" * 60)

    original_claim = "N_min ~ 5-10 for real embeddings"
    models_where_5_is_stable = sum(1 for r in model_results if r.N_min <= 10)
    models_total = len(model_results)

    print(f"\nOriginal claim: {original_claim}")
    print(f"Models where N<=10 is stable: {models_where_5_is_stable}/{models_total}")

    if models_where_5_is_stable == models_total:
        claim_verdict = "SUPPORTED"
        claim_explanation = "All tested models achieve CV<0.10 at N<=10"
    elif models_where_5_is_stable >= models_total * 0.8:
        claim_verdict = "MOSTLY_SUPPORTED"
        claim_explanation = f"{models_where_5_is_stable}/{models_total} models achieve CV<0.10 at N<=10"
    elif models_where_5_is_stable >= models_total * 0.5:
        claim_verdict = "PARTIALLY_SUPPORTED"
        claim_explanation = f"Only {models_where_5_is_stable}/{models_total} models achieve CV<0.10 at N<=10"
    else:
        claim_verdict = "NOT_SUPPORTED"
        claim_explanation = f"Only {models_where_5_is_stable}/{models_total} models achieve CV<0.10 at N<=10"

    print(f"Claim verdict: {claim_verdict}")
    print(f"Explanation: {claim_explanation}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n1. Scaling Law: {verdict}")
    print(f"   - Best fit: {best_fit} (R^2 = {best_r2:.4f})")
    print(f"   - N_min range across models: {N_min_range[0]} to {N_min_range[1]}")

    print(f"\n2. Original Claim (N=5-10 sufficient): {claim_verdict}")

    print(f"\n3. Practical Recommendation:")
    if verdict == "NO_UNIVERSAL_SCALING" or N_min_cv > 0.30:
        print("   - N_min is MODEL-SPECIFIC, not universal")
        print(f"   - Test your specific model; don't assume N=5-10 works")
        print(f"   - Safe default: N >= {int(np.max(N_min_values))}")
    else:
        print(f"   - Use N >= {int(np.mean(N_min_values) + np.std(N_min_values))}")

    # Build results dictionary
    results = {
        "test_id": "Q26_RIGOROUS_SCALING_TEST",
        "timestamp": str(np.datetime64('now')),
        "corpus_size": len(texts),
        "n_trials": n_trials,
        "models_tested": [r.model_name for r in model_results],
        "N_candidates": N_candidates,
        "results_by_model": {
            r.model_name: {
                "D": r.D,
                "N_min": r.N_min,
                "stability_curve": {str(k): v for k, v in r.stability_curve.items()}
            }
            for r in model_results
        },
        "scaling_analysis": {
            "D_values": D_values.tolist(),
            "N_min_values": N_min_values.tolist(),
            "fits": scaling_fits,
            "best_fit": best_fit,
            "best_r2": best_r2,
            "N_min_cv": N_min_cv,
            "N_min_range": list(N_min_range)
        },
        "verdicts": {
            "scaling_law": verdict,
            "scaling_explanation": explanation,
            "original_claim": original_claim,
            "claim_verdict": claim_verdict,
            "claim_explanation": claim_explanation
        }
    }

    return results


def main():
    """Run the test and save results."""
    results = run_comprehensive_test()

    if results is None:
        print("\nTest failed - no results to save")
        return

    # Save results
    output_path = Path(__file__).parent / "q26_scaling_test_results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")

    print("\n" + "=" * 80)
    print(f"FINAL VERDICT: {results['verdicts']['scaling_law']}")
    print(f"CLAIM STATUS: {results['verdicts']['claim_verdict']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
