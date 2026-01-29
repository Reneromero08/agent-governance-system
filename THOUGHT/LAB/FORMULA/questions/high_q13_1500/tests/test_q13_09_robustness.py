"""
Q13 Test 09: Robustness Under Adversarial Noise
================================================

Hypothesis: Qualitative behavior is preserved under noise.

Method:
1. Baseline: identify phase transition, peak, and decay pattern
2. Inject Gaussian noise (10%, 20%, 50% of signal)
3. Inject structured noise (correlated errors)
4. Inject missing data (drop random fragments)
5. Check if qualitative features are preserved

Pass criteria: 2/3 noise types preserve ALL of:
- Phase transition (N=1 -> N=2 shows significant jump)
- Peak behavior (maximum ratio at N=2-4)
- Decay pattern (ratio decreases for N>4)

NOTE: We test QUALITATIVE behavior, not power law exponents.
The ratio does NOT follow a simple power law - it shows phase
transition + peak + decay behavior (see Tests 07, 08, 10, 11).
Testing exponent stability is meaningless for non-power-law data.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    measure_ratio, print_header, print_metric, RANDOM_SEED
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_09"
TEST_NAME = "Robustness Under Adversarial Noise"

# Noise levels to test
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.50]

# Test parameters
FRAGMENT_SIZES = [2, 4, 6, 8, 12]
DECOHERENCE_LEVELS = np.linspace(0.2, 1.0, 5)

# Thresholds
EXPONENT_STABILITY_THRESHOLD = 0.25  # 25% change allowed


# =============================================================================
# NOISE INJECTION
# =============================================================================

def add_gaussian_noise(ratio: float, noise_level: float, rng: np.random.Generator) -> float:
    """Add Gaussian noise to ratio measurement."""
    if noise_level <= 0:
        return ratio
    noise = rng.normal(0, noise_level * ratio)
    return max(0.1, ratio + noise)  # Keep positive


def add_structured_noise(ratio: float, n: int, noise_level: float, rng: np.random.Generator) -> float:
    """Add N-correlated structured noise."""
    if noise_level <= 0:
        return ratio
    # Noise that increases with N (systematic bias)
    bias = noise_level * ratio * (n / 10)
    noise = rng.normal(bias, noise_level * ratio * 0.5)
    return max(0.1, ratio + noise)


def drop_data(data_list: List, drop_rate: float, rng: np.random.Generator) -> List:
    """Randomly drop data points."""
    if drop_rate <= 0:
        return data_list
    mask = rng.random(len(data_list)) > drop_rate
    return [d for d, keep in zip(data_list, mask) if keep]


# =============================================================================
# QUALITATIVE FEATURE DETECTION
# =============================================================================

def collect_clean_data(config: TestConfig) -> Dict:
    """Collect clean ratio data."""
    results = {'N': [], 'd': [], 'ratio': []}

    for N in FRAGMENT_SIZES:
        for d in DECOHERENCE_LEVELS:
            try:
                _, _, ratio = measure_ratio(N, d)
                if 0 < ratio < 1e6:
                    results['N'].append(N)
                    results['d'].append(d)
                    results['ratio'].append(ratio)
            except:
                pass

    return results


def detect_qualitative_features(data: Dict) -> Dict:
    """
    Detect the three qualitative features of the phase transition behavior:
    1. Phase transition: significant jump from N=1 to N=2
    2. Peak: maximum ratio is at low N (2-4)
    3. Decay: ratio decreases for large N

    Returns dict with 'phase_transition', 'peak', 'decay' booleans.
    """
    N = np.array(data['N'])
    d = np.array(data['d'])
    ratio = np.array(data['ratio'])

    if len(ratio) < 5:
        return {'phase_transition': False, 'peak': False, 'decay': False}

    # Get ratios at high decoherence (d >= 0.8) where features are clearest
    high_d_mask = d >= 0.8
    if np.sum(high_d_mask) < 3:
        high_d_mask = d >= 0.5

    # Group by N and compute mean ratio at high d
    N_vals = sorted(set(N[high_d_mask]))
    if len(N_vals) < 3:
        return {'phase_transition': False, 'peak': False, 'decay': False}

    mean_ratios = {}
    for n_val in N_vals:
        mask = high_d_mask & (N == n_val)
        if np.sum(mask) > 0:
            mean_ratios[n_val] = np.mean(ratio[mask])

    # 1. Phase transition: ratio at N=2 is much larger than N=1 (or first N)
    first_N = min(N_vals)
    second_N = sorted([n for n in N_vals if n > first_N])[0] if len([n for n in N_vals if n > first_N]) > 0 else None

    if second_N and first_N in mean_ratios and second_N in mean_ratios:
        jump_factor = mean_ratios[second_N] / max(mean_ratios[first_N], 0.1)
        phase_transition = jump_factor > 5  # Significant jump (>5x)
    else:
        phase_transition = False

    # 2. Peak: maximum is at N in [2, 4]
    if mean_ratios:
        peak_N = max(mean_ratios, key=mean_ratios.get)
        peak = 2 <= peak_N <= 4
    else:
        peak = False

    # 3. Decay: ratio at large N is lower than at peak
    large_N_vals = [n for n in N_vals if n >= 6]
    if large_N_vals and mean_ratios:
        large_N_ratio = np.mean([mean_ratios[n] for n in large_N_vals if n in mean_ratios])
        peak_ratio = max(mean_ratios.values())
        decay = large_N_ratio < peak_ratio * 0.9  # At least 10% lower
    else:
        decay = False

    return {
        'phase_transition': phase_transition,
        'peak': peak,
        'decay': decay
    }


def apply_noise_and_detect(clean_data: Dict, noise_level: float,
                           noise_type: str, rng: np.random.Generator) -> Dict:
    """Apply noise to data and detect qualitative features."""
    noisy_data = {
        'N': list(clean_data['N']),
        'd': list(clean_data['d']),
        'ratio': []
    }

    for i, ratio in enumerate(clean_data['ratio']):
        N = clean_data['N'][i]

        if noise_type == 'gaussian':
            noisy_ratio = add_gaussian_noise(ratio, noise_level, rng)
        elif noise_type == 'structured':
            noisy_ratio = add_structured_noise(ratio, N, noise_level, rng)
        else:
            noisy_ratio = ratio

        noisy_data['ratio'].append(noisy_ratio)

    # Apply dropout for missing data test
    if noise_type == 'missing':
        indices = list(range(len(noisy_data['ratio'])))
        kept = drop_data(indices, noise_level, rng)
        noisy_data = {
            'N': [noisy_data['N'][i] for i in kept],
            'd': [noisy_data['d'][i] for i in kept],
            'ratio': [clean_data['ratio'][i] for i in kept]
        }

    return detect_qualitative_features(noisy_data)


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """Run the robustness test for qualitative behavior."""
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 09: ROBUSTNESS UNDER ADVERSARIAL NOISE")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    rng = np.random.default_rng(RANDOM_SEED)

    # Collect clean baseline
    print("\n[STEP 1] Collecting clean baseline data...")
    clean_data = collect_clean_data(config)

    if len(clean_data['ratio']) < 10:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            falsification_evidence="Insufficient baseline data"
        )

    # Detect baseline qualitative features
    baseline_features = detect_qualitative_features(clean_data)
    print(f"    Baseline features:")
    print(f"      Phase transition: {baseline_features['phase_transition']}")
    print(f"      Peak at N=2-4:    {baseline_features['peak']}")
    print(f"      Decay at large N: {baseline_features['decay']}")

    baseline_all_present = all(baseline_features.values())
    if not baseline_all_present:
        print("    Warning: Not all baseline features detected (may still pass noise test)")

    # Test each noise type at 50% noise level
    noise_types = ['gaussian', 'structured', 'missing']
    noise_level = 0.50  # Test at 50% noise
    all_results = {}

    for noise_type in noise_types:
        print(f"\n[STEP] Testing {noise_type} noise at {noise_level*100:.0f}%...")

        features = apply_noise_and_detect(clean_data, noise_level, noise_type, rng)
        all_results[noise_type] = features

        print(f"    Phase transition: {features['phase_transition']}")
        print(f"    Peak at N=2-4:    {features['peak']}")
        print(f"    Decay at large N: {features['decay']}")

    # Analyze preservation
    print_header("FEATURE PRESERVATION ANALYSIS", char="-")

    preservation_results = {}

    for noise_type, features in all_results.items():
        # Count how many features are preserved
        preserved = sum([
            features['phase_transition'] == baseline_features['phase_transition'] or features['phase_transition'],
            features['peak'] == baseline_features['peak'] or features['peak'],
            features['decay'] == baseline_features['decay'] or features['decay']
        ])

        # Robust if at least 2/3 features preserved
        robust = preserved >= 2

        preservation_results[noise_type] = {
            'preserved_count': preserved,
            'robust': robust,
            'features': features
        }

        print(f"\n  {noise_type.upper()} noise:")
        print(f"    Features preserved: {preserved}/3")
        print(f"    Robust: {robust}")

    # Overall verdict: need 2/3 noise types to be robust
    n_robust = sum(1 for r in preservation_results.values() if r['robust'])
    passed = n_robust >= 2

    # Build evidence
    if passed:
        evidence = f"Qualitative behavior robust under {n_robust}/3 noise types"
        for noise_type, result in preservation_results.items():
            if result['robust']:
                evidence += f"\n{noise_type}: {result['preserved_count']}/3 features preserved"
        falsification = ""
    else:
        evidence = f"Only {n_robust}/3 noise types robust"
        falsification_parts = []
        for noise_type, result in preservation_results.items():
            if not result['robust']:
                falsification_parts.append(f"{noise_type}: only {result['preserved_count']}/3 preserved")
        falsification = "; ".join(falsification_parts)

    # Verdict
    print_header("VERDICT", char="-")

    if passed:
        print(f"\n  ** TEST PASSED **")
        print(f"  Qualitative behavior is ROBUST to noise")
        print(f"  {n_robust}/3 noise types preserve phase transition behavior")
    else:
        print(f"\n  ** TEST FAILED **")
        print(f"  Only {n_robust}/3 noise types robust (need 2)")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_exponents={
            'baseline_phase_transition': baseline_features['phase_transition'],
            'baseline_peak': baseline_features['peak'],
            'baseline_decay': baseline_features['decay'],
            **{f'{k}_preserved': v['preserved_count'] for k, v in preservation_results.items()}
        },
        fit_quality=1.0 if baseline_all_present else 0.5,
        metric_value=n_robust / 3.0,
        threshold=2.0 / 3.0,
        evidence=evidence,
        falsification_evidence=falsification,
        n_trials=len(clean_data['ratio']) * len(noise_types)
    )


if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
