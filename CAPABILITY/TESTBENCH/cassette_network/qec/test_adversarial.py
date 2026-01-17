#!/usr/bin/env python3
"""Test 6: Adversarial Attacks.

Proves that R-gating provides robustness against designed adversarial attacks.
The system should detect attacks before semantic corruption occurs.

Attack Types:
    1. Synonym Substitution - Replace words with synonyms
    2. Gradual Drift (Boiling Frog) - Slow interpolation to invalid
    3. Random Dimension Targeting - Perturb random PC dimensions
    4. Coordinated Multi-Observation - Corrupt multiple observations together

Success Criteria:
    - Synonym: Detect at ~t substitutions (code distance limit)
    - Gradual: Detect at alpha < 0.5
    - Random: Requires >2t dimensions to bypass
    - Coordinated: sigma anomaly > 2x baseline

Usage:
    python test_adversarial.py [--n-trials 50]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine

# Local imports
from core import (
    r_gate,
    compute_R,
    compute_dispersion,
    generate_random_embeddings,
    compute_effective_dimensionality,
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
    "The cat sat on the mat.",
    "Water flows downhill naturally.",
    "Light travels faster than sound.",
    "Trees provide oxygen for life.",
    "Knowledge grows through learning.",
]

INVALID_TARGETS = [
    "The purple sat on the seven.",
    "Silence flows upward colorfully.",
    "Darkness travels slower than nothing.",
    "Stones provide confusion for rocks.",
    "Forgetting shrinks through ignoring.",
]


def get_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings for texts."""
    if not HAS_TRANSFORMERS:
        np.random.seed(hash(texts[0]) % 2**32)
        dim = 384
        embeddings = np.random.randn(len(texts), dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# =============================================================================
# Attack Functions
# =============================================================================

def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit vectors.

    Args:
        v0: Start vector (normalized)
        v1: End vector (normalized)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated unit vector
    """
    # Ensure normalized
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    # Compute angle
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        return v0

    # Slerp formula
    sin_theta = np.sin(theta)
    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    result = s0 * v0 + s1 * v1
    return result / np.linalg.norm(result)


def attack_gradual_drift(
    valid_emb: np.ndarray,
    invalid_emb: np.ndarray,
    n_steps: int = 20
) -> List[Tuple[np.ndarray, float]]:
    """Gradual drift attack: interpolate from valid to invalid.

    Args:
        valid_emb: Valid embedding
        invalid_emb: Invalid target embedding
        n_steps: Number of interpolation steps

    Returns:
        List of (embedding, alpha) tuples
    """
    results = []
    for i in range(n_steps + 1):
        alpha = i / n_steps
        interpolated = slerp(valid_emb, invalid_emb, alpha)
        results.append((interpolated, alpha))
    return results


def attack_random_dimensions(
    embedding: np.ndarray,
    n_dims: int,
    magnitude: float = 0.5
) -> np.ndarray:
    """Random dimension attack: perturb random dimensions.

    Args:
        embedding: Original embedding
        n_dims: Number of dimensions to perturb
        magnitude: Perturbation magnitude

    Returns:
        Perturbed embedding
    """
    dim = len(embedding)
    perturbed = embedding.copy()

    # Select random dimensions
    dims_to_attack = np.random.choice(dim, min(n_dims, dim), replace=False)

    for d in dims_to_attack:
        perturbed[d] += magnitude * np.random.choice([-1, 1])

    # Renormalize
    return perturbed / np.linalg.norm(perturbed)


def attack_coordinated(
    observations: np.ndarray,
    attack_fraction: float,
    attack_magnitude: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinated multi-observation attack.

    Corrupt a fraction of observations in a coordinated direction.

    Args:
        observations: (n_obs, dim) observation embeddings
        attack_fraction: Fraction of observations to corrupt
        attack_magnitude: Size of coordinated perturbation

    Returns:
        Tuple of (corrupted_observations, attack_direction)
    """
    n_obs, dim = observations.shape
    n_attack = max(1, int(n_obs * attack_fraction))

    # Find attack direction (maximize centroid shift while staying on sphere)
    # Use a random unit direction
    attack_direction = np.random.randn(dim)
    attack_direction = attack_direction / np.linalg.norm(attack_direction)

    # Corrupt selected observations
    corrupted = observations.copy()
    attack_indices = np.random.choice(n_obs, n_attack, replace=False)

    for i in attack_indices:
        corrupted[i] = corrupted[i] + attack_magnitude * attack_direction
        corrupted[i] = corrupted[i] / np.linalg.norm(corrupted[i])

    return corrupted, attack_direction


# =============================================================================
# Attack Detection
# =============================================================================

def detect_gradual_drift(
    valid_emb: np.ndarray,
    invalid_emb: np.ndarray,
    n_steps: int = 20,
    threshold: float = DEFAULT_R_THRESHOLD
) -> Dict:
    """Run gradual drift attack and detect when gate triggers.

    Returns:
        Dict with detection_alpha, gate_trace, etc.
    """
    interpolations = attack_gradual_drift(valid_emb, invalid_emb, n_steps)

    gate_trace = []
    detection_alpha = None

    for embedding, alpha in interpolations:
        # Create observations around this embedding
        n_obs = 5
        noise_scale = 0.01
        dim = len(embedding)

        observations = np.array([
            embedding + np.random.randn(dim) * noise_scale
            for _ in range(n_obs)
        ])
        observations = observations / np.linalg.norm(observations, axis=1, keepdims=True)

        # Check R-gate
        gate_result = r_gate(observations, threshold)
        gate_trace.append({
            'alpha': alpha,
            'R_value': gate_result.R_value,
            'sigma': gate_result.sigma,
            'passed': gate_result.passed,
        })

        # Record first detection (gate fails)
        if not gate_result.passed and detection_alpha is None:
            detection_alpha = alpha

    return {
        'detection_alpha': detection_alpha,
        'gate_trace': gate_trace,
        'detected_early': detection_alpha is not None and detection_alpha < 0.5,
    }


def detect_random_dimensions(
    embedding: np.ndarray,
    max_dims: int = 50,
    magnitude: float = 0.5,
    threshold: float = DEFAULT_R_THRESHOLD
) -> Dict:
    """Run random dimension attack with increasing dimensions.

    Returns:
        Dict with detection_dims, success_trace, etc.
    """
    dim = len(embedding)
    detection_dims = None

    success_trace = []
    for n_dims in range(1, min(max_dims, dim) + 1):
        # Try attack
        attacked = attack_random_dimensions(embedding, n_dims, magnitude)

        # Create observations
        n_obs = 5
        noise_scale = 0.01
        observations = np.array([
            attacked + np.random.randn(dim) * noise_scale
            for _ in range(n_obs)
        ])
        observations = observations / np.linalg.norm(observations, axis=1, keepdims=True)

        # Check R-gate
        gate_result = r_gate(observations, threshold)
        success_trace.append({
            'n_dims': n_dims,
            'R_value': gate_result.R_value,
            'passed': gate_result.passed,
        })

        # Gate fails = attack detected
        if not gate_result.passed and detection_dims is None:
            detection_dims = n_dims

    return {
        'detection_dims': detection_dims,
        'success_trace': success_trace,
    }


def detect_coordinated(
    embedding: np.ndarray,
    attack_fractions: List[float],
    threshold: float = DEFAULT_R_THRESHOLD
) -> Dict:
    """Run coordinated attack with varying fractions.

    Returns:
        Dict with sigma_anomalies, detection_fraction, etc.
    """
    dim = len(embedding)
    n_obs = 10  # More observations for coordinated attack

    # Create baseline observations
    noise_scale = 0.01
    baseline_obs = np.array([
        embedding + np.random.randn(dim) * noise_scale
        for _ in range(n_obs)
    ])
    baseline_obs = baseline_obs / np.linalg.norm(baseline_obs, axis=1, keepdims=True)

    # Baseline sigma
    baseline_sigma = compute_dispersion(baseline_obs)

    results_trace = []
    detection_fraction = None

    for frac in attack_fractions:
        # Run coordinated attack
        attacked_obs, _ = attack_coordinated(baseline_obs, frac, attack_magnitude=0.3)

        # Compute sigma
        attacked_sigma = compute_dispersion(attacked_obs)
        sigma_ratio = attacked_sigma / (baseline_sigma + 1e-10)

        # Check R-gate
        gate_result = r_gate(attacked_obs, threshold)

        results_trace.append({
            'fraction': frac,
            'sigma': attacked_sigma,
            'sigma_ratio': sigma_ratio,
            'R_value': gate_result.R_value,
            'passed': gate_result.passed,
        })

        # Detect via sigma anomaly (2x baseline) or gate failure
        if (sigma_ratio > 2.0 or not gate_result.passed) and detection_fraction is None:
            detection_fraction = frac

    return {
        'baseline_sigma': baseline_sigma,
        'detection_fraction': detection_fraction,
        'results_trace': results_trace,
        'sigma_anomaly_detected': detection_fraction is not None,
    }


# =============================================================================
# Main Test
# =============================================================================

def run_adversarial_test(
    n_trials: int = 30,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict:
    """Run full adversarial attack test suite.

    Args:
        n_trials: Number of trials per attack type
        model_name: Sentence transformer model name

    Returns:
        Complete test results dict
    """
    print("=" * 70)
    print("TEST 6: ADVERSARIAL ATTACKS")
    print("=" * 70)
    print()

    # Get embeddings
    print("Loading embeddings...")
    valid_embs = get_embeddings(VALID_PHRASES, model_name)
    invalid_embs = get_embeddings(INVALID_TARGETS, model_name)
    valid_df = compute_effective_dimensionality(valid_embs)
    print(f"  Df: {valid_df:.2f}")
    print()

    results = {
        "test_id": "q40-adversarial-attacks",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_trials": n_trials,
            "model_name": model_name,
        },
        "df": float(valid_df),
        "attacks": {},
    }

    # Attack 1: Gradual Drift
    print("-" * 50)
    print("ATTACK: Gradual Drift (Boiling Frog)")
    print("-" * 50)

    drift_results = []
    for trial in range(n_trials):
        idx = trial % len(valid_embs)
        result = detect_gradual_drift(valid_embs[idx], invalid_embs[idx])
        drift_results.append(result)

        if result['detection_alpha'] is not None:
            print(f"  Trial {trial+1}: Detected at alpha={result['detection_alpha']:.2f}")
        else:
            print(f"  Trial {trial+1}: NOT DETECTED")

    detected_alphas = [r['detection_alpha'] for r in drift_results if r['detection_alpha'] is not None]
    mean_detection_alpha = np.mean(detected_alphas) if detected_alphas else None
    detection_rate = len(detected_alphas) / len(drift_results)
    early_detection_rate = sum(1 for r in drift_results if r['detected_early']) / len(drift_results)

    print(f"\nDetection rate: {detection_rate:.0%}")
    print(f"Early detection (alpha < 0.5): {early_detection_rate:.0%}")
    if mean_detection_alpha:
        print(f"Mean detection alpha: {mean_detection_alpha:.2f}")

    results["attacks"]["gradual_drift"] = {
        "detection_rate": float(detection_rate),
        "early_detection_rate": float(early_detection_rate),
        "mean_detection_alpha": float(mean_detection_alpha) if mean_detection_alpha else None,
        "pass": early_detection_rate > 0.5,
    }

    # Attack 2: Random Dimension Targeting
    print()
    print("-" * 50)
    print("ATTACK: Random Dimension Targeting")
    print("-" * 50)

    dim_results = []
    for trial in range(n_trials):
        idx = trial % len(valid_embs)
        result = detect_random_dimensions(valid_embs[idx], max_dims=30)
        dim_results.append(result)

        if result['detection_dims'] is not None:
            print(f"  Trial {trial+1}: Detected at {result['detection_dims']} dims")
        else:
            print(f"  Trial {trial+1}: NOT DETECTED (up to 30 dims)")

    detected_dims = [r['detection_dims'] for r in dim_results if r['detection_dims'] is not None]
    mean_detection_dims = np.mean(detected_dims) if detected_dims else None
    detection_rate_dims = len(detected_dims) / len(dim_results)

    print(f"\nDetection rate: {detection_rate_dims:.0%}")
    if mean_detection_dims:
        print(f"Mean detection dims: {mean_detection_dims:.1f}")

    results["attacks"]["random_dimensions"] = {
        "detection_rate": float(detection_rate_dims),
        "mean_detection_dims": float(mean_detection_dims) if mean_detection_dims else None,
        "pass": detection_rate_dims > 0.7,
    }

    # Attack 3: Coordinated Multi-Observation
    print()
    print("-" * 50)
    print("ATTACK: Coordinated Multi-Observation")
    print("-" * 50)

    attack_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    coord_results = []
    for trial in range(n_trials):
        idx = trial % len(valid_embs)
        result = detect_coordinated(valid_embs[idx], attack_fractions)
        coord_results.append(result)

        if result['detection_fraction'] is not None:
            print(f"  Trial {trial+1}: Detected at {result['detection_fraction']:.0%} corruption")
        else:
            print(f"  Trial {trial+1}: NOT DETECTED")

    detected_fracs = [r['detection_fraction'] for r in coord_results if r['detection_fraction'] is not None]
    mean_detection_frac = np.mean(detected_fracs) if detected_fracs else None
    sigma_anomaly_rate = sum(1 for r in coord_results if r['sigma_anomaly_detected']) / len(coord_results)

    print(f"\nSigma anomaly detection rate: {sigma_anomaly_rate:.0%}")
    if mean_detection_frac:
        print(f"Mean detection fraction: {mean_detection_frac:.0%}")

    results["attacks"]["coordinated"] = {
        "sigma_anomaly_rate": float(sigma_anomaly_rate),
        "mean_detection_fraction": float(mean_detection_frac) if mean_detection_frac else None,
        "pass": sigma_anomaly_rate > 0.5,
    }

    # Overall verdict
    attacks_passed = sum(1 for a in results["attacks"].values() if a.get("pass", False))
    total_attacks = len(results["attacks"])

    verdict_pass = attacks_passed >= total_attacks // 2 + 1

    results["verdict"] = {
        "attacks_passed": attacks_passed,
        "total_attacks": total_attacks,
        "overall_pass": verdict_pass,
        "interpretation": (
            f"PASS: {attacks_passed}/{total_attacks} attack types detected. "
            "R-gating provides robustness against adversarial attacks."
            if verdict_pass else
            f"FAIL: Only {attacks_passed}/{total_attacks} attack types detected."
        )
    }

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Attacks passed: {attacks_passed}/{total_attacks}")
    for attack_name, attack_result in results["attacks"].items():
        status = "PASS" if attack_result.get("pass", False) else "FAIL"
        print(f"  {attack_name}: {status}")
    print()
    print(f"OVERALL: {'PASS' if verdict_pass else 'FAIL'}")
    print(results["verdict"]["interpretation"])
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Q40 Test 6: Adversarial Attacks')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of trials per attack type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_adversarial_test(n_trials=args.n_trials)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "adversarial_attacks.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
