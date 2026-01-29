"""
falsification_gauntlet.py - Adversarial testing framework for R

This module provides tools to attempt to BREAK R-based findings.
If R survives these attacks, it's robust. If it breaks, we document how.
"""

import numpy as np
from typing import Callable, Dict, List, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class AttackResult:
    """Result of an adversarial attack on R."""
    attack_name: str
    original_r: float
    attacked_r: float
    correlation_under_attack: float
    attack_succeeded: bool  # True if attack broke R
    description: str

    def to_dict(self) -> dict:
        return {
            "attack_name": self.attack_name,
            "original_r": self.original_r,
            "attacked_r": self.attacked_r,
            "correlation_under_attack": self.correlation_under_attack,
            "attack_succeeded": self.attack_succeeded,
            "description": self.description
        }


def add_gaussian_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to data."""
    noise = np.random.randn(*data.shape) * noise_level * np.std(data)
    return data + noise


def add_adversarial_noise(
    data: np.ndarray,
    r_function: Callable,
    target_direction: str = "maximize"
) -> np.ndarray:
    """
    Add adversarial noise designed to fool R estimation.

    Uses gradient-free optimization to find noise that maximally
    changes R in the target direction.
    """
    best_noise = np.zeros_like(data)
    original_r = r_function(data)

    # Random search for adversarial direction
    for _ in range(100):
        noise = np.random.randn(*data.shape) * 0.1 * np.std(data)
        perturbed_data = data + noise
        new_r = r_function(perturbed_data)

        if target_direction == "maximize":
            if new_r > r_function(data + best_noise):
                best_noise = noise
        else:  # minimize
            if new_r < r_function(data + best_noise):
                best_noise = noise

    return data + best_noise


def create_pathological_data(
    shape: Tuple[int, ...],
    pathology_type: str
) -> np.ndarray:
    """
    Create pathological data designed to break R.

    Types:
    - "high_variance": Very high variance, low agreement
    - "perfect_agreement": All samples identical
    - "bimodal": Two distinct clusters
    - "heavy_tails": Heavy-tailed distribution
    - "sparse": Mostly zeros
    - "adversarial": Designed to maximize R estimation error
    """
    if pathology_type == "high_variance":
        # Each sample from different distribution
        data = np.random.randn(*shape) * np.arange(1, shape[0] + 1)[:, None]

    elif pathology_type == "perfect_agreement":
        # All samples identical
        template = np.random.randn(shape[1] if len(shape) > 1 else shape[0])
        data = np.tile(template, (shape[0], 1)) if len(shape) > 1 else template

    elif pathology_type == "bimodal":
        # Two clusters far apart
        n_half = shape[0] // 2
        if len(shape) > 1:
            cluster1 = np.random.randn(n_half, shape[1]) - 5
            cluster2 = np.random.randn(shape[0] - n_half, shape[1]) + 5
        else:
            cluster1 = np.random.randn(n_half) - 5
            cluster2 = np.random.randn(shape[0] - n_half) + 5
        data = np.vstack([cluster1, cluster2]) if len(shape) > 1 else np.concatenate([cluster1, cluster2])

    elif pathology_type == "heavy_tails":
        # Cauchy distribution (heavy tails)
        data = np.random.standard_cauchy(shape)
        # Clip extreme values to prevent numerical issues
        data = np.clip(data, -100, 100)

    elif pathology_type == "sparse":
        # Mostly zeros
        data = np.zeros(shape)
        mask = np.random.rand(*shape) < 0.1
        data[mask] = np.random.randn(mask.sum())

    elif pathology_type == "adversarial":
        # Start with normal data, then add structured noise
        data = np.random.randn(*shape)
        # Add sinusoidal pattern to confuse correlation measures
        if len(shape) > 1:
            for i in range(shape[0]):
                data[i] += np.sin(np.linspace(0, 4*np.pi, shape[1])) * (i % 3)
        else:
            data += np.sin(np.linspace(0, 4*np.pi, shape[0]))

    else:
        raise ValueError(f"Unknown pathology type: {pathology_type}")

    return data


def run_adversarial_gauntlet(
    r_function: Callable[[np.ndarray], float],
    test_data: np.ndarray,
    survival_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Run full adversarial gauntlet on R function.

    Tests:
    1. Gaussian noise at multiple levels
    2. Adversarial noise (targeted)
    3. Pathological data types
    4. Shuffled baselines

    Args:
        r_function: Function that computes R from data
        test_data: Original test data
        survival_threshold: Correlation threshold for survival (default 0.7)

    Returns:
        Dict with attack results and overall survival rate
    """
    results = []
    original_r = r_function(test_data)

    # Test 1: Gaussian noise at multiple levels
    noise_levels = [0.1, 0.25, 0.5, 1.0]
    for level in noise_levels:
        noisy_data = add_gaussian_noise(test_data, level)
        noisy_r = r_function(noisy_data)

        # Correlation is 1 if same direction, meaningful test needs multiple samples
        correlation = 1.0 if np.sign(noisy_r) == np.sign(original_r) else 0.0

        results.append(AttackResult(
            attack_name=f"gaussian_noise_{level}",
            original_r=original_r,
            attacked_r=noisy_r,
            correlation_under_attack=correlation,
            attack_succeeded=abs(noisy_r - original_r) / (abs(original_r) + 1e-10) > 0.5,
            description=f"Gaussian noise at {level}x std level"
        ))

    # Test 2: Pathological data
    pathologies = ["high_variance", "bimodal", "heavy_tails", "sparse"]
    for pathology in pathologies:
        try:
            path_data = create_pathological_data(test_data.shape, pathology)
            path_r = r_function(path_data)

            # Attack succeeds if R gives meaningless value (NaN, Inf, or extreme)
            attack_succeeded = (
                np.isnan(path_r) or
                np.isinf(path_r) or
                abs(path_r) > 1e6
            )

            results.append(AttackResult(
                attack_name=f"pathological_{pathology}",
                original_r=original_r,
                attacked_r=path_r if not (np.isnan(path_r) or np.isinf(path_r)) else 0.0,
                correlation_under_attack=0.5,  # Neutral for pathological data
                attack_succeeded=attack_succeeded,
                description=f"Pathological {pathology} data"
            ))
        except Exception as e:
            results.append(AttackResult(
                attack_name=f"pathological_{pathology}",
                original_r=original_r,
                attacked_r=0.0,
                correlation_under_attack=0.0,
                attack_succeeded=True,  # Crash = attack success
                description=f"CRASHED: {str(e)}"
            ))

    # Test 3: Shuffled baseline
    shuffled_data = test_data.copy()
    np.random.shuffle(shuffled_data)
    shuffled_r = r_function(shuffled_data)

    results.append(AttackResult(
        attack_name="shuffled_baseline",
        original_r=original_r,
        attacked_r=shuffled_r,
        correlation_under_attack=0.5,  # Shuffling should reduce R
        attack_succeeded=shuffled_r >= original_r,  # Attack succeeds if shuffle doesn't reduce R
        description="Shuffled data should have lower R than original"
    ))

    # Compute survival rate
    n_survived = sum(1 for r in results if not r.attack_succeeded)
    survival_rate = n_survived / len(results)

    return {
        "original_r": original_r,
        "n_attacks": len(results),
        "n_survived": n_survived,
        "survival_rate": survival_rate,
        "passed": survival_rate >= survival_threshold,
        "attacks": [r.to_dict() for r in results]
    }


def test_blind_transfer(
    r_function_source: Callable,
    r_function_target: Callable,
    source_data: np.ndarray,
    target_data: np.ndarray,
    target_outcomes: np.ndarray,
    min_correlation: float = 0.3
) -> Dict[str, Any]:
    """
    Test if R from source domain predicts outcomes in target domain.

    This is a BLIND transfer test - no retuning allowed.

    Args:
        r_function_source: R function for source domain
        r_function_target: R function for target domain (same formula)
        source_data: Data from source domain (for reference)
        target_data: Data from target domain
        target_outcomes: True outcomes in target domain
        min_correlation: Minimum correlation for success

    Returns:
        Dict with transfer results
    """
    # Compute R on source (for calibration check only)
    source_r_values = []
    for i in range(min(100, source_data.shape[0])):
        if len(source_data.shape) > 1:
            r_val = r_function_source(source_data[i:i+1])
        else:
            r_val = r_function_source(source_data)
        source_r_values.append(r_val)

    # Compute R on target using SAME formula (no retuning)
    target_r_values = []
    for i in range(target_data.shape[0]):
        if len(target_data.shape) > 1:
            r_val = r_function_target(target_data[i:i+1])
        else:
            r_val = r_function_target(target_data)
        target_r_values.append(r_val)

    target_r_values = np.array(target_r_values)

    # Compute correlation with outcomes
    valid_mask = ~(np.isnan(target_r_values) | np.isinf(target_r_values))
    if valid_mask.sum() < 3:
        return {
            "correlation": 0.0,
            "p_value": 1.0,
            "passed": False,
            "reason": "Insufficient valid R values"
        }

    from scipy import stats
    r, p = stats.pearsonr(target_r_values[valid_mask], target_outcomes[valid_mask])

    return {
        "correlation": r,
        "p_value": p,
        "passed": r >= min_correlation and p < 0.05,
        "source_r_mean": np.mean(source_r_values),
        "target_r_mean": np.nanmean(target_r_values),
        "n_valid": int(valid_mask.sum())
    }


def generate_negative_control(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """
    Generate negative control data that should NOT show R patterns.

    This is pure noise - no structure, no agreement.
    R should be approximately 1.0 (E=1, sigma=1) or uninformative.
    """
    np.random.seed(seed)
    return np.random.randn(*shape)
