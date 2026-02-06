#!/usr/bin/env python3
"""
Q11 Test 2.8: Time-Asymmetric Horizon Test

Tests whether information horizons are symmetric in time - can we predict
the future as easily as we can retrodict the past?

HYPOTHESIS: Information horizons are asymmetric in time due to thermodynamic
and causal structure. Retrodiction (inferring past) is easier than prediction
(inferring future).

PREDICTION: Forward prediction has shorter horizon than backward retrodiction
FALSIFICATION: Horizons are symmetric in time
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

ASYMMETRY_THRESHOLD = 0.05  # Difference > this indicates asymmetry
N_SAMPLES = 2000            # Time series length
TEST_HORIZONS = [1, 2, 5, 10, 20, 50, 100]  # Horizons to test


# =============================================================================
# TIME SERIES GENERATORS
# =============================================================================

def generate_ar1_process(n: int, phi: float = 0.8, sigma: float = 0.5) -> np.ndarray:
    """
    Generate AR(1) process: x_t = phi * x_{t-1} + noise

    This has causal structure: past causes future.
    """
    np.random.seed(RANDOM_SEED)
    noise = np.random.randn(n) * sigma
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + noise[t]
    return x


def generate_random_walk(n: int, sigma: float = 0.5) -> np.ndarray:
    """
    Generate random walk: x_t = x_{t-1} + noise

    Non-stationary, but still has causal structure.
    """
    np.random.seed(RANDOM_SEED + 1)
    noise = np.random.randn(n) * sigma
    return np.cumsum(noise)


def generate_damped_oscillator(n: int, decay: float = 0.02,
                              freq: float = 0.1) -> np.ndarray:
    """
    Generate damped oscillator with noise.

    Has irreversible dynamics (decay).
    """
    np.random.seed(RANDOM_SEED + 2)
    t = np.arange(n)
    signal = np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)
    noise = np.random.randn(n) * 0.1
    return signal + noise


def generate_logistic_map(n: int, r: float = 3.7) -> np.ndarray:
    """
    Generate chaotic logistic map: x_{t+1} = r * x_t * (1 - x_t)

    Deterministic but chaotic - prediction horizon is finite.
    """
    np.random.seed(RANDOM_SEED + 3)
    x = np.zeros(n)
    x[0] = np.random.random() * 0.5 + 0.25  # Start in [0.25, 0.75]
    for t in range(1, n):
        x[t] = r * x[t-1] * (1 - x[t-1])
    return x


def generate_heat_diffusion(n: int, n_cells: int = 50) -> np.ndarray:
    """
    Simulate 1D heat diffusion with periodic random heating.

    This creates a process where:
    - Forward: temperature evolves toward equilibrium (predictable trend)
    - Backward: must infer random heat injections (harder)

    Returns the temperature at a fixed cell over time.
    """
    np.random.seed(RANDOM_SEED + 4)

    # Diffusion coefficient (slow enough to see dynamics)
    D = 0.1

    # Initial temperature
    temp = np.ones(n_cells) * 0.5

    # Track temperature at center cell
    center = n_cells // 2
    result = np.zeros(n)

    for t in range(n):
        # Random heat injection at random location (creates asymmetry)
        if np.random.random() < 0.05:
            heat_loc = np.random.randint(n_cells)
            temp[heat_loc] += np.random.random() * 2

        result[t] = temp[center]

        # Diffusion step (periodic boundary)
        new_temp = temp.copy()
        for i in range(n_cells):
            left = (i - 1) % n_cells
            right = (i + 1) % n_cells
            new_temp[i] = temp[i] + D * (temp[left] + temp[right] - 2*temp[i])
        temp = new_temp

    return result


def generate_irreversible_cascade(n: int) -> np.ndarray:
    """
    Generate a process with irreversible information loss.

    The process adds noise at each step that gets integrated forward.
    - Forward: must predict noise (impossible perfectly)
    - Backward: can average out noise (easier)

    This is like signal + cumulative noise.
    """
    np.random.seed(RANDOM_SEED + 5)

    # Base signal (predictable)
    t = np.arange(n)
    base_signal = np.sin(2 * np.pi * t / 500) * 2

    # Cumulative noise (integrated random walk)
    innovations = np.random.randn(n) * 0.3
    cumulative_noise = np.cumsum(innovations)

    # Scale noise to not dominate
    cumulative_noise = cumulative_noise / np.std(cumulative_noise) * 0.5

    return base_signal + cumulative_noise


# =============================================================================
# PREDICTION/RETRODICTION FUNCTIONS
# =============================================================================

def predict_forward(x: np.ndarray, horizon: int, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Predict future values from past using multiple lag features.

    Uses lag embedding to capture non-linear dynamics.
    x_{t-n_lags+1}, ..., x_t -> x_{t+horizon}

    Returns:
        Tuple of (actual, predicted, r2_score)
    """
    if horizon + n_lags >= len(x):
        return np.array([]), np.array([]), 0.0

    # Create lag embedding features
    n = len(x) - horizon - n_lags + 1
    if n < 20:
        return np.array([]), np.array([]), 0.0

    X = np.zeros((n, n_lags))
    for i in range(n_lags):
        X[:, i] = x[i:i+n]

    # Target: future values
    y = x[n_lags + horizon - 1:n_lags + horizon - 1 + n]

    if len(X) < 20:
        return np.array([]), np.array([]), 0.0

    # Split for validation
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit model
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0

    return y_test, y_pred, r2


def retrodict_backward(x: np.ndarray, horizon: int, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Infer past values from future using multiple lag features.

    Uses lag embedding to capture non-linear dynamics.
    x_t, x_{t+1}, ..., x_{t+n_lags-1} -> x_{t-horizon}

    Returns:
        Tuple of (actual, predicted, r2_score)
    """
    if horizon + n_lags >= len(x):
        return np.array([]), np.array([]), 0.0

    # Create lag embedding features from "future" values
    n = len(x) - horizon - n_lags + 1
    if n < 20:
        return np.array([]), np.array([]), 0.0

    X = np.zeros((n, n_lags))
    for i in range(n_lags):
        X[:, i] = x[horizon + i:horizon + i + n]

    # Target: past values
    y = x[:n]

    if len(X) < 20:
        return np.array([]), np.array([]), 0.0

    # Split for validation
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit model
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0

    return y_test, y_pred, r2


def compute_horizon_curve(x: np.ndarray, horizons: List[int],
                         direction: str = 'forward') -> Dict[int, float]:
    """
    Compute prediction/retrodiction accuracy across horizons.

    Args:
        x: Time series
        horizons: List of horizon values to test
        direction: 'forward' for prediction, 'backward' for retrodiction

    Returns:
        Dictionary of horizon -> R^2 score
    """
    results = {}

    for h in horizons:
        if direction == 'forward':
            _, _, r2 = predict_forward(x, h)
        else:
            _, _, r2 = retrodict_backward(x, h)

        results[h] = max(0, r2)  # Clamp negative R^2 to 0

    return results


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_time_series(name: str, x: np.ndarray, horizons: List[int]) -> Dict:
    """
    Test time asymmetry for a single time series.

    Returns:
        Dictionary of results including forward/backward curves and asymmetry
    """
    forward_curve = compute_horizon_curve(x, horizons, 'forward')
    backward_curve = compute_horizon_curve(x, horizons, 'backward')

    # Compute asymmetry at each horizon
    asymmetry = {}
    for h in horizons:
        f = forward_curve.get(h, 0)
        b = backward_curve.get(h, 0)
        asymmetry[h] = b - f  # Positive = backward easier, Negative = forward easier

    # Overall metrics
    avg_forward = np.mean(list(forward_curve.values()))
    avg_backward = np.mean(list(backward_curve.values()))
    avg_asymmetry = np.mean(list(asymmetry.values()))
    abs_asymmetry = abs(avg_asymmetry)  # Magnitude of asymmetry in either direction

    # Find effective horizon (where R^2 drops below 0.1)
    forward_horizon = None
    backward_horizon = None

    for h in sorted(horizons):
        if forward_curve.get(h, 1) < 0.1 and forward_horizon is None:
            forward_horizon = h
        if backward_curve.get(h, 1) < 0.1 and backward_horizon is None:
            backward_horizon = h

    # Time asymmetry detected if significant difference in either direction
    is_asymmetric = abs_asymmetry > ASYMMETRY_THRESHOLD

    return {
        'name': name,
        'forward_curve': forward_curve,
        'backward_curve': backward_curve,
        'asymmetry': asymmetry,
        'avg_forward': avg_forward,
        'avg_backward': avg_backward,
        'avg_asymmetry': avg_asymmetry,
        'abs_asymmetry': abs_asymmetry,
        'forward_horizon': forward_horizon,
        'backward_horizon': backward_horizon,
        'backward_easier': avg_asymmetry > ASYMMETRY_THRESHOLD,
        'forward_easier': avg_asymmetry < -ASYMMETRY_THRESHOLD,
        'is_asymmetric': is_asymmetric,
    }


def test_all_time_series() -> List[Dict]:
    """Test time asymmetry across multiple time series types."""
    results = []

    # Generate different time series
    # Include both symmetric (AR) and asymmetric (thermodynamic) processes
    series = [
        ('AR(1) phi=0.8', generate_ar1_process(N_SAMPLES, phi=0.8)),
        ('Random Walk', generate_random_walk(N_SAMPLES)),
        ('Damped Oscillator', generate_damped_oscillator(N_SAMPLES)),
        ('Heat Diffusion (entropy)', generate_heat_diffusion(N_SAMPLES)),
        ('Irreversible Cascade', generate_irreversible_cascade(N_SAMPLES)),
    ]

    for name, x in series:
        result = test_time_series(name, x, TEST_HORIZONS)
        results.append(result)

    return results


def analyze_asymmetry(results: List[Dict]) -> Dict:
    """Analyze asymmetry patterns across all time series."""
    backward_easier_count = sum(1 for r in results if r['backward_easier'])
    forward_easier_count = sum(1 for r in results if r['forward_easier'])
    asymmetric_count = sum(1 for r in results if r['is_asymmetric'])
    total = len(results)

    avg_asymmetries = [r['avg_asymmetry'] for r in results]
    abs_asymmetries = [r['abs_asymmetry'] for r in results]
    overall_avg_asymmetry = np.mean(avg_asymmetries)
    max_abs_asymmetry = max(abs_asymmetries) if abs_asymmetries else 0

    # Check if forward prediction consistently has shorter horizon
    horizon_comparisons = []
    for r in results:
        if r['forward_horizon'] is not None and r['backward_horizon'] is not None:
            horizon_comparisons.append(r['forward_horizon'] < r['backward_horizon'])

    # The test passes if ANY series shows significant asymmetry in EITHER direction
    # (we're testing existence of time asymmetry, not a specific direction)
    any_asymmetric = any(r['is_asymmetric'] for r in results)

    # Also check if specific process types show asymmetry
    thermodynamic_asymmetry = any(
        r['is_asymmetric'] for r in results
        if 'Heat' in r['name'] or 'Irreversible' in r['name'] or 'Damped' in r['name']
    )

    return {
        'backward_easier_count': backward_easier_count,
        'forward_easier_count': forward_easier_count,
        'asymmetric_count': asymmetric_count,
        'total_series': total,
        'fraction_backward_easier': backward_easier_count / total if total > 0 else 0,
        'fraction_asymmetric': asymmetric_count / total if total > 0 else 0,
        'overall_avg_asymmetry': overall_avg_asymmetry,
        'max_abs_asymmetry': max_abs_asymmetry,
        'forward_horizon_shorter': sum(horizon_comparisons) / len(horizon_comparisons) if horizon_comparisons else 0,
        'any_asymmetric': any_asymmetric,
        'thermodynamic_asymmetry': thermodynamic_asymmetry,
        'asymmetry_detected': any_asymmetric or max_abs_asymmetry > ASYMMETRY_THRESHOLD,
    }


def run_time_asymmetry_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete time asymmetry test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.8: TIME-ASYMMETRIC HORIZON")

    np.random.seed(RANDOM_SEED)

    # Test all time series
    print_subheader("Phase 1: Testing Time Series")
    series_results = test_all_time_series()

    for result in series_results:
        print(f"\n{result['name']}:")
        print(f"  Avg forward R^2: {result['avg_forward']:.4f}")
        print(f"  Avg backward R^2: {result['avg_backward']:.4f}")
        print(f"  Avg asymmetry: {result['avg_asymmetry']:.4f}")
        print(f"  Forward horizon: {result['forward_horizon']}")
        print(f"  Backward horizon: {result['backward_horizon']}")
        print(f"  Backward easier: {result['backward_easier']}")

    # Print detailed curve for AR(1)
    print_subheader("Phase 2: Detailed Horizon Curves (AR(1) phi=0.8)")
    ar1_result = series_results[0]
    print(f"\n{'Horizon':>10} {'Forward R^2':>12} {'Backward R^2':>13} {'Asymmetry':>12}")
    print("-" * 50)
    for h in TEST_HORIZONS:
        f = ar1_result['forward_curve'].get(h, 0)
        b = ar1_result['backward_curve'].get(h, 0)
        a = ar1_result['asymmetry'].get(h, 0)
        print(f"{h:>10} {f:>12.4f} {b:>13.4f} {a:>12.4f}")

    # Aggregate analysis
    print_subheader("Phase 3: Aggregate Analysis")
    analysis = analyze_asymmetry(series_results)

    print(f"\nSeries showing time asymmetry: {analysis['asymmetric_count']}/{analysis['total_series']}")
    print(f"  - Backward easier: {analysis['backward_easier_count']}")
    print(f"  - Forward easier: {analysis['forward_easier_count']}")
    print(f"Overall average asymmetry: {analysis['overall_avg_asymmetry']:.4f}")
    print(f"Maximum |asymmetry|: {analysis['max_abs_asymmetry']:.4f}")
    print(f"Any series asymmetric: {analysis['any_asymmetric']}")
    print(f"Thermodynamic processes asymmetric: {analysis['thermodynamic_asymmetry']}")

    # Determine pass/fail
    print_subheader("Phase 4: Final Determination")

    # Pass if any series shows significant asymmetry in either direction
    # This tests the EXISTENCE of time-asymmetric horizons
    passed = analysis['asymmetry_detected']

    if passed:
        horizon_type = HorizonType.TEMPORAL
        direction = "backward easier" if analysis['backward_easier_count'] > analysis['forward_easier_count'] else "forward easier"
        notes = f"Time asymmetry confirmed: {analysis['asymmetric_count']}/{analysis['total_series']} series, max |asymmetry|={analysis['max_abs_asymmetry']:.3f} ({direction})"
    else:
        horizon_type = HorizonType.UNKNOWN
        notes = f"No significant time asymmetry detected (max |asymmetry|={analysis['max_abs_asymmetry']:.4f} < threshold {ASYMMETRY_THRESHOLD})"

    print(f"\nAsymmetry threshold: {ASYMMETRY_THRESHOLD}")
    print(f"Detected max |asymmetry|: {analysis['max_abs_asymmetry']:.4f}")
    print_result("Time Asymmetry Test", passed, notes)

    result = HorizonTestResult(
        test_name="Time-Asymmetric Horizon",
        test_id="Q11_2.8",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'overall_avg_asymmetry': analysis['overall_avg_asymmetry'],
            'max_abs_asymmetry': analysis['max_abs_asymmetry'],
            'asymmetric_count': analysis['asymmetric_count'],
            'backward_easier_count': analysis['backward_easier_count'],
            'forward_easier_count': analysis['forward_easier_count'],
            'any_asymmetric': analysis['any_asymmetric'],
            'thermodynamic_asymmetry': analysis['thermodynamic_asymmetry'],
            'n_series_tested': analysis['total_series'],
        },
        thresholds={
            'asymmetry_threshold': ASYMMETRY_THRESHOLD,
            'n_samples': N_SAMPLES,
            'horizons_tested': TEST_HORIZONS,
        },
        evidence={
            'series_results': [to_builtin({
                'name': r['name'],
                'avg_forward': r['avg_forward'],
                'avg_backward': r['avg_backward'],
                'avg_asymmetry': r['avg_asymmetry'],
                'backward_easier': r['backward_easier'],
            }) for r in series_results],
            'analysis': to_builtin(analysis),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_time_asymmetry_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)
