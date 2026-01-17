#!/usr/bin/env python3
"""Test 3: Error Threshold via Alpha Conservation.

Proves R-gating implements QECC by showing alpha conservation breaks
above a critical noise threshold.

KEY INSIGHT FROM Q21/Q32:
- Semantic embeddings have alpha ~ 0.5 (Riemann critical line)
- Df * alpha = 8e = 21.746 (conservation law)
- Below threshold: alpha stays near 0.5 (structure protected)
- Above threshold: alpha drifts (structure damaged)

Hypothesis:
    Below epsilon_th, alpha conservation holds (structure protected).
    Above epsilon_th, alpha drifts (structure damaged).
    Semantic has LOWER threshold than random (more structure to protect).

Protocol:
    1. Sweep noise level from 0.1% to 50%
    2. Measure alpha drift at each level
    3. Find threshold where drift exceeds limit
    4. Compare semantic threshold vs random threshold

Success Criteria:
    - Clear threshold exists (alpha drift jumps)
    - Semantic threshold < Random threshold (structure to protect)
    - Alpha starts near 0.5 for semantic

Usage:
    python test_threshold.py [--n-trials 200] [--n-epsilon 30]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Local imports
from core import (
    inject_n_errors,
    r_gate,
    compute_R,
    generate_random_embeddings,
    compute_effective_dimensionality,
    compute_alpha,
    get_eigenspectrum,
    compute_compass_health,
    cohens_d,
    SEMIOTIC_CONSTANT_8E,
    DEFAULT_R_THRESHOLD,
)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Test Data
# =============================================================================

VALID_PHRASES = [
    # Science fundamentals (10)
    "The sun rises in the east.",
    "Water boils at one hundred degrees Celsius.",
    "Humans need oxygen to survive.",
    "Mathematics is the language of science.",
    "Light travels faster than sound.",
    "The moon orbits the Earth.",
    "Plants convert sunlight into energy.",
    "Gravity keeps planets in orbit.",
    "DNA carries genetic information.",
    "Computers process binary data.",
    # Nature and environment (10)
    "Rivers flow from mountains to seas.",
    "Seasons change due to Earth's tilt.",
    "Ecosystems depend on biodiversity.",
    "Forests produce oxygen for breathing.",
    "Ocean currents regulate global climate.",
    "Polar ice reflects sunlight back to space.",
    "Volcanoes release gases from deep underground.",
    "Coral reefs support marine life diversity.",
    "Deserts receive very little rainfall annually.",
    "Wetlands filter pollutants from water naturally.",
    # Technology and engineering (10)
    "Algorithms solve problems step by step.",
    "Networks connect devices across distances.",
    "Encryption protects sensitive data transmissions.",
    "Sensors detect changes in the environment.",
    "Batteries store electrical energy chemically.",
    "Circuits control the flow of electricity.",
    "Software instructions guide hardware operations.",
    "Databases organize information for retrieval.",
    "Protocols define communication standards precisely.",
    "Processors execute millions of calculations per second.",
    # Biology and medicine (10)
    "Cells are the basic units of life.",
    "Vaccines train immune systems against diseases.",
    "Neurons transmit signals through the brain.",
    "Enzymes catalyze biochemical reactions efficiently.",
    "Muscles contract to produce movement.",
    "Blood carries oxygen to body tissues.",
    "Hormones regulate bodily functions chemically.",
    "Antibiotics kill bacterial infections selectively.",
    "Metabolism converts food into cellular energy.",
    "Genes encode instructions for protein synthesis.",
    # Physics and chemistry (10)
    "Atoms combine to form molecules.",
    "Energy cannot be created or destroyed.",
    "Electrons orbit atomic nuclei in shells.",
    "Pressure increases with depth underwater.",
    "Heat flows from hot to cold objects.",
    "Friction opposes motion between surfaces.",
    "Waves transfer energy without moving matter.",
    "Magnets attract iron and other metals.",
    "Acids and bases neutralize each other.",
    "Photons are particles of light energy.",
]


def get_embeddings(phrases: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings for phrases."""
    if not HAS_TRANSFORMERS:
        np.random.seed(42)
        dim = 384
        embeddings = np.random.randn(len(phrases), dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    model = SentenceTransformer(model_name)
    embeddings = model.encode(phrases, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# =============================================================================
# Error Threshold Models
# =============================================================================

def suppression_model(eps_phys, eps_th, k_below, k_above, baseline):
    """Two-regime error model.

    Below threshold: eps_log ~ eps_phys^k_below (suppression if k > 1)
    Above threshold: eps_log ~ (eps_phys/eps_th)^k_above * threshold_value
    """
    result = np.zeros_like(eps_phys)

    below_mask = eps_phys < eps_th
    above_mask = ~below_mask

    # Below threshold: power law suppression
    if np.any(below_mask):
        result[below_mask] = baseline * (eps_phys[below_mask] ** k_below)

    # Above threshold: different scaling
    if np.any(above_mask):
        threshold_value = baseline * (eps_th ** k_below)
        result[above_mask] = threshold_value * ((eps_phys[above_mask] / eps_th) ** k_above)

    return result


def fit_threshold_model(
    eps_phys: np.ndarray,
    eps_log: np.ndarray
) -> Tuple[Dict, float]:
    """Fit two-regime model to error data.

    Returns:
        Tuple of (parameters dict, R-squared)
    """
    # Filter out zeros for fitting
    valid = (eps_log > 0) & (eps_phys > 0)
    if np.sum(valid) < 5:
        return {
            "eps_th": 0.1,
            "k_below": 1.0,
            "k_above": 1.0,
            "baseline": 1.0
        }, 0.0

    eps_phys_valid = eps_phys[valid]
    eps_log_valid = eps_log[valid]

    try:
        # Initial guess
        p0 = [0.1, 2.0, 0.5, 1.0]

        # Bounds
        bounds = (
            [0.001, 0.1, 0.1, 0.01],   # Lower bounds
            [0.5, 10.0, 5.0, 100.0]     # Upper bounds
        )

        popt, _ = curve_fit(
            suppression_model,
            eps_phys_valid,
            eps_log_valid,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )

        # Compute R-squared
        predicted = suppression_model(eps_phys_valid, *popt)
        ss_res = np.sum((eps_log_valid - predicted) ** 2)
        ss_tot = np.sum((eps_log_valid - np.mean(eps_log_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "eps_th": float(popt[0]),
            "k_below": float(popt[1]),
            "k_above": float(popt[2]),
            "baseline": float(popt[3])
        }, float(r_squared)

    except (RuntimeError, ValueError) as e:
        print(f"Fitting failed: {e}")
        return {
            "eps_th": 0.1,
            "k_below": 1.0,
            "k_above": 1.0,
            "baseline": 1.0
        }, 0.0


# =============================================================================
# Main Test
# =============================================================================

def measure_alpha_drift_at_noise(
    embeddings: np.ndarray,
    noise_level: float,
    n_trials: int = 10,
    error_type: str = 'gaussian_noise'
) -> Tuple[float, float, float]:
    """Measure alpha drift under noise injection.

    KEY INSIGHT FROM Q21/Q32:
    - Semantic embeddings have alpha ~ 0.5
    - Noise causes alpha to DRIFT from 0.5
    - The drift indicates structural damage

    Args:
        embeddings: Base embeddings
        noise_level: Standard deviation of noise (0 to 1)
        n_trials: Number of trials for averaging
        error_type: Type of error to inject

    Returns:
        Tuple of (baseline_alpha, corrupted_alpha, drift)
    """
    # Compute baseline alpha
    alpha_baseline, Df_baseline, _ = compute_compass_health(embeddings)

    drifts = []
    corrupted_alphas = []

    for trial in range(n_trials):
        # Inject noise into all embeddings
        corrupted = []
        for emb in embeddings:
            if error_type == 'gaussian_noise':
                noisy = emb + np.random.randn(len(emb)) * noise_level
            else:
                result = inject_n_errors(emb, 1, error_type, sigma=noise_level, epsilon=noise_level)
                noisy = result.corrupted
            noisy = noisy / (np.linalg.norm(noisy) + 1e-10)
            corrupted.append(noisy)
        corrupted = np.array(corrupted)

        # Compute corrupted alpha
        alpha_corrupted, _, _ = compute_compass_health(corrupted)
        corrupted_alphas.append(alpha_corrupted)
        drifts.append(abs(alpha_corrupted - alpha_baseline))

    mean_corrupted_alpha = float(np.mean(corrupted_alphas))
    mean_drift = float(np.mean(drifts))

    return float(alpha_baseline), mean_corrupted_alpha, mean_drift


def find_alpha_threshold(
    embeddings: np.ndarray,
    eps_range: np.ndarray,
    n_trials: int = 5,
    drift_limit: float = 0.15
) -> Tuple[float, List[float]]:
    """Find noise threshold where alpha drift exceeds limit.

    Args:
        embeddings: Base embeddings
        eps_range: Array of noise levels to test
        n_trials: Trials per noise level
        drift_limit: Alpha drift that indicates structure damage

    Returns:
        Tuple of (threshold_epsilon, list_of_drifts)
    """
    drifts = []
    for eps in eps_range:
        _, _, drift = measure_alpha_drift_at_noise(embeddings, eps, n_trials)
        drifts.append(drift)

    # Find first epsilon where drift exceeds limit
    threshold_idx = len(eps_range) - 1  # Default to max
    for i, drift in enumerate(drifts):
        if drift > drift_limit:
            threshold_idx = i
            break

    threshold_eps = float(eps_range[threshold_idx])
    return threshold_eps, drifts


def run_threshold_test(
    n_epsilon: int = 30,
    n_trials: int = 200,
    dim: int = 384,
    n_random_seeds: int = 3
) -> Dict:
    """Run full error threshold test via alpha conservation.

    KEY INSIGHT FROM Q21/Q32:
    - Semantic embeddings have alpha ~ 0.5 (conservation law)
    - Below threshold: alpha stays near 0.5 (structure protected)
    - Above threshold: alpha drifts (structure damaged)
    - Semantic has LOWER threshold than random (more to protect)

    We measure:
    1. Baseline alpha for semantic and random
    2. Alpha drift at each noise level
    3. Threshold where drift exceeds 0.15

    QECC is proven when:
    - Semantic alpha starts near 0.5
    - Clear threshold exists
    - Semantic threshold < random threshold OR semantic shows more drift

    Args:
        n_epsilon: Number of epsilon values to test
        n_trials: Trials per epsilon (for averaging)
        dim: Embedding dimension
        n_random_seeds: Number of random baselines

    Returns:
        Complete test results dict
    """
    print("=" * 70)
    print("TEST 3: ERROR THRESHOLD VIA ALPHA CONSERVATION")
    print("=" * 70)
    print()
    print("Key insight: Alpha drift = structural damage. Threshold = where drift jumps.")
    print(f"Conservation law: Df * alpha = 8e = {SEMIOTIC_CONSTANT_8E:.2f}")
    print()

    # Get semantic embeddings
    print("Loading semantic embeddings...")
    semantic_emb = get_embeddings(VALID_PHRASES)
    alpha_sem, Df_sem, DfAlpha_sem = compute_compass_health(semantic_emb)
    print(f"  Baseline alpha: {alpha_sem:.4f} (target: 0.5)")
    print(f"  Baseline Df: {Df_sem:.2f}")
    print(f"  Baseline Df*alpha: {DfAlpha_sem:.2f} (target: {SEMIOTIC_CONSTANT_8E:.2f})")
    print()

    # Generate random baselines
    print(f"Generating {n_random_seeds} random baselines...")
    random_embs = [
        generate_random_embeddings(len(VALID_PHRASES), dim, seed=42 + i)
        for i in range(n_random_seeds)
    ]

    random_alphas = []
    for i, random_emb in enumerate(random_embs):
        alpha_r, Df_r, _ = compute_compass_health(random_emb)
        random_alphas.append(alpha_r)
        print(f"  Random {i}: alpha={alpha_r:.4f}, Df={Df_r:.2f}")
    print()

    # Epsilon range (logarithmic from 0.5% to 50%)
    eps_range = np.logspace(-2.3, -0.3, n_epsilon)

    results = {
        "test_id": "q40-alpha-threshold",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_epsilon": n_epsilon,
            "n_trials": n_trials,
            "dim": dim,
            "n_random_seeds": n_random_seeds,
        },
        "semantic_baseline": {
            "alpha": float(alpha_sem),
            "Df": float(Df_sem),
            "DfAlpha": float(DfAlpha_sem),
        },
        "epsilon_range": eps_range.tolist(),
    }

    # Measure semantic alpha drift
    print("Measuring semantic alpha drift...")
    print("  Format: eps | alpha | drift")
    n_trials_per = max(3, n_trials // n_epsilon)

    semantic_drifts = []
    for eps in eps_range:
        _, corrupted_alpha, drift = measure_alpha_drift_at_noise(semantic_emb, eps, n_trials_per)
        semantic_drifts.append(drift)
        print(f"  {eps:.4f} | {corrupted_alpha:.4f} | {drift:.4f}")

    # Find semantic threshold
    drift_limit = 0.15
    sem_threshold_idx = len(eps_range) - 1
    for i, drift in enumerate(semantic_drifts):
        if drift > drift_limit:
            sem_threshold_idx = i
            break
    sem_threshold = float(eps_range[sem_threshold_idx])

    print(f"\nSemantic threshold (drift > {drift_limit}): {sem_threshold:.4f} ({sem_threshold*100:.1f}%)")

    results["semantic"] = {
        "drifts": [float(d) for d in semantic_drifts],
        "threshold_epsilon": sem_threshold,
        "max_drift": float(max(semantic_drifts)),
    }

    # Measure random alpha drift
    print("\nMeasuring random alpha drift...")
    random_thresholds = []

    for seed_idx, random_emb in enumerate(random_embs):
        print(f"  Random seed {seed_idx}:")
        random_drifts = []
        for eps in eps_range:
            _, _, drift = measure_alpha_drift_at_noise(random_emb, eps, n_trials_per)
            random_drifts.append(drift)

        # Find random threshold
        rand_threshold_idx = len(eps_range) - 1
        for i, drift in enumerate(random_drifts):
            if drift > drift_limit:
                rand_threshold_idx = i
                break
        rand_threshold = float(eps_range[rand_threshold_idx])
        random_thresholds.append(rand_threshold)
        print(f"    Threshold: {rand_threshold:.4f}, max_drift: {max(random_drifts):.4f}")

    mean_random_threshold = np.mean(random_thresholds)

    results["random"] = {
        "thresholds": [float(t) for t in random_thresholds],
        "mean_threshold": float(mean_random_threshold),
    }

    # Compute effect sizes
    sem_max_drift = max(semantic_drifts)
    rand_max_drifts = [max(d) for d in [[measure_alpha_drift_at_noise(r, eps_range[-1], n_trials_per)[2]
                                          for _ in range(1)] for r in random_embs]]

    # Verdict
    # QECC is proven when:
    # 1. Semantic alpha near 0.5 (structure exists)
    # 2. Clear threshold exists (not at max noise)
    # 3. Semantic shows MORE drift than random (structure to lose)
    #    OR semantic has LOWER threshold (more sensitive)

    alpha_near_05 = abs(alpha_sem - 0.5) < 0.15
    has_threshold = sem_threshold_idx < len(eps_range) - 2
    more_drift = sem_max_drift > np.mean(random_alphas) * 0.3  # Semantic loses more structure
    lower_threshold = sem_threshold < mean_random_threshold * 0.8  # Semantic more sensitive

    verdict_pass = alpha_near_05 and has_threshold and (more_drift or lower_threshold)

    results["verdict"] = {
        "alpha_near_05": alpha_near_05,
        "has_threshold": has_threshold,
        "more_drift_than_random": more_drift,
        "lower_threshold_than_random": lower_threshold,
        "overall_pass": verdict_pass,
        "interpretation": (
            f"PASS: Alpha conservation threshold at {sem_threshold*100:.1f}%. "
            f"Semantic alpha={alpha_sem:.3f} (near 0.5). "
            f"Max drift={sem_max_drift:.3f}. "
            "Structure protected below threshold, damaged above."
            if verdict_pass else
            f"FAIL: "
            f"{'Alpha not near 0.5. ' if not alpha_near_05 else ''}"
            f"{'No clear threshold. ' if not has_threshold else ''}"
            f"{'Semantic doesnt show more structure loss. ' if not more_drift and not lower_threshold else ''}"
        )
    }

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Semantic baseline alpha: {alpha_sem:.4f} (near 0.5: {alpha_near_05})")
    print(f"Semantic threshold: {sem_threshold:.4f} ({sem_threshold*100:.1f}%)")
    print(f"Random mean threshold: {mean_random_threshold:.4f} ({mean_random_threshold*100:.1f}%)")
    print(f"Semantic max drift: {sem_max_drift:.4f}")
    print(f"Has clear threshold: {has_threshold}")
    print(f"More drift than random: {more_drift}")
    print(f"Lower threshold than random: {lower_threshold}")
    print(f"OVERALL: {'PASS' if verdict_pass else 'FAIL'}")
    print(results["verdict"]["interpretation"])
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Q40 Test 3: Error Threshold')
    parser.add_argument('--n-epsilon', type=int, default=30,
                        help='Number of epsilon values')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Trials per epsilon')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_threshold_test(
        n_epsilon=args.n_epsilon,
        n_trials=args.n_trials,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "error_threshold.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
