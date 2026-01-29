"""
Q13: The 36x Ratio - Shared Utilities
=====================================

Provides shared infrastructure for all Q13 scaling law tests.

Key components:
- ScalingLawResult: Standardized result dataclass
- ScalingLaw: Enumeration of scaling law types
- Quantum state creation and measurement functions
- Statistical analysis utilities
- Model fitting and comparison tools

Author: AGS Research
Date: 2026-01-19
"""

import sys
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
N_TRIALS = 100  # Minimum trials per data point
CONFIDENCE_LEVEL = 0.99  # 99% CI
PASS_THRESHOLD = 10  # Need 10+ of 12 tests to pass

# Fragment counts for scaling analysis
FRAGMENT_COUNTS = [1, 2, 4, 6, 8, 12, 16, 24, 32]

# Decoherence levels for sweep
DECOHERENCE_LEVELS = np.linspace(0.0, 1.0, 21)  # 21 steps including endpoints


# =============================================================================
# ENUMS
# =============================================================================

class ScalingLaw(Enum):
    """Types of scaling laws to test."""
    POWER = "power"           # C * N^alpha * d^beta
    EXPONENTIAL = "exponential"  # C * exp(alpha * N * d)
    LOGARITHMIC = "logarithmic"  # C * (1 + alpha * log(N)) * d^beta
    LINEAR = "linear"         # C * (1 + alpha * N) * d
    QUADRATIC = "quadratic"   # C * (1 + alpha*N + beta*N^2) * d
    CRITICAL = "critical"     # C * |d - d_c|^(-gamma) * N^alpha
    UNKNOWN = "unknown"


class TestStatus(Enum):
    """Test outcome status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class ScalingLawResult:
    """Result from a single Q13 test."""
    test_name: str
    test_id: str
    passed: bool
    scaling_law: str = "unknown"
    scaling_exponents: Dict[str, float] = field(default_factory=dict)
    fit_quality: float = 0.0  # R^2 or similar
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metric_value: float = 0.0
    threshold: float = 0.0
    evidence: str = ""
    falsification_evidence: str = ""
    bayes_factor: Optional[float] = None
    n_trials: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TestConfig:
    """Configuration for Q13 tests."""
    seed: int = RANDOM_SEED
    n_trials: int = N_TRIALS
    fragment_counts: List[int] = field(default_factory=lambda: list(FRAGMENT_COUNTS))
    decoherence_levels: np.ndarray = field(default_factory=lambda: DECOHERENCE_LEVELS.copy())
    verbose: bool = True
    confidence_level: float = CONFIDENCE_LEVEL


# =============================================================================
# QUANTUM UTILITIES (from quantum_darwinism_test_v2.py)
# =============================================================================

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False


def create_qd_state(n_fragments: int, decoherence: float):
    """
    Create a proper quantum Darwinism state.

    At decoherence=0: |+>|0...0> (system in superposition, env unentangled)
    At decoherence=1: (|0>|0...0> + |1>|1...1>)/sqrt(2) (GHZ-like, full redundancy)
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("QuTiP required for quantum tests")

    up = qt.basis(2, 0)
    down = qt.basis(2, 1)

    if n_fragments == 0:
        if decoherence < 0.01:
            return (up + down).unit()
        else:
            return (np.sqrt(1-decoherence) * (up + down).unit() +
                   np.sqrt(decoherence) * up)

    if decoherence < 0.01:
        # Pure superposition, environment unentangled
        sys = (up + down).unit()
        env = up
        for _ in range(n_fragments - 1):
            env = qt.tensor(env, up)
        return qt.tensor(sys, env)

    elif decoherence > 0.99:
        # Full GHZ-like state (perfect redundancy)
        branch_0 = up
        branch_1 = down
        for _ in range(n_fragments):
            branch_0 = qt.tensor(branch_0, up)
            branch_1 = qt.tensor(branch_1, down)
        return (branch_0 + branch_1).unit()

    else:
        # Partial decoherence - interpolate
        env_0 = up
        for _ in range(n_fragments - 1):
            env_0 = qt.tensor(env_0, up)
        branch_0 = qt.tensor(up, env_0)

        d = decoherence
        env_1_single = (np.sqrt(1-d) * up + np.sqrt(d) * down).unit()
        env_1 = env_1_single
        for _ in range(n_fragments - 1):
            env_1 = qt.tensor(env_1, env_1_single)
        branch_1 = qt.tensor(down, env_1)

        return (branch_0 + branch_1).unit()


def get_fragment_probs(state, frag_indices: List[int]) -> np.ndarray:
    """Get probability distribution for fragment(s)."""
    rho = state.ptrace(frag_indices)
    probs = np.abs(np.diag(rho.full()))
    probs = probs / probs.sum()
    return probs


# =============================================================================
# THE FORMULA
# =============================================================================

def compute_essence(probs: np.ndarray) -> float:
    """Distance from uniform distribution."""
    uniform = np.ones_like(probs) / len(probs)
    return max(0.01, np.sqrt(np.sum((probs - uniform) ** 2)))


def compute_grad_S(probs_list: List[np.ndarray]) -> float:
    """Dispersion across multiple observations."""
    if len(probs_list) < 2:
        return 0.01
    arr = np.array(probs_list)
    dispersion = np.mean(np.var(arr, axis=0))
    return max(0.01, dispersion)


def compute_R(probs_list: List[np.ndarray], sigma: float = 0.5, Df: float = 1.0) -> float:
    """
    Compute R = (E / grad_S) * sigma^Df
    """
    if len(probs_list) == 0:
        return 0.0
    mean_probs = np.mean(probs_list, axis=0)
    E = compute_essence(mean_probs)
    grad_S = compute_grad_S(probs_list)
    return (E / grad_S) * (sigma ** Df)


def measure_ratio(n_fragments: int, decoherence: float) -> Tuple[float, float, float]:
    """
    Measure the context improvement ratio for given parameters.

    Returns:
        (R_single, R_joint, ratio)
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("QuTiP required")

    if n_fragments < 1:
        return (1.0, 1.0, 1.0)

    state = create_qd_state(n_fragments, decoherence)

    # R for single fragment
    single_probs = get_fragment_probs(state, [1])
    R_single = compute_R([single_probs], sigma=0.5, Df=1.0)

    # R for joint observation (all fragments together)
    joint_indices = list(range(1, n_fragments + 1))
    joint_probs = get_fragment_probs(state, joint_indices)
    Df_joint = np.log(n_fragments + 1)
    R_joint = compute_R([joint_probs], sigma=0.5, Df=Df_joint)

    # Compute ratio
    ratio = R_joint / max(R_single, 0.001)

    return (R_single, R_joint, ratio)


# =============================================================================
# SCALING LAW MODELS
# =============================================================================

def power_law(N: np.ndarray, d: np.ndarray, C: float, alpha: float, beta: float) -> np.ndarray:
    """Power law: Ratio = C * N^alpha * d^beta"""
    # Handle d=0 case
    d_safe = np.maximum(d, 0.001)
    return C * np.power(N, alpha) * np.power(d_safe, beta)


def exponential_law(N: np.ndarray, d: np.ndarray, C: float, alpha: float) -> np.ndarray:
    """Exponential: Ratio = C * exp(alpha * N * d)"""
    return C * np.exp(alpha * N * d)


def logarithmic_law(N: np.ndarray, d: np.ndarray, C: float, alpha: float, beta: float) -> np.ndarray:
    """Logarithmic: Ratio = C * (1 + alpha * log(N)) * d^beta"""
    d_safe = np.maximum(d, 0.001)
    return C * (1 + alpha * np.log(np.maximum(N, 1))) * np.power(d_safe, beta)


def linear_law(N: np.ndarray, d: np.ndarray, C: float, alpha: float) -> np.ndarray:
    """Linear: Ratio = C * (1 + alpha * N) * d"""
    return C * (1 + alpha * N) * d


def quadratic_law(N: np.ndarray, d: np.ndarray, C: float, alpha: float, beta: float) -> np.ndarray:
    """Quadratic: Ratio = C * (1 + alpha*N + beta*N^2) * d"""
    return C * (1 + alpha * N + beta * N * N) * d


def critical_law(N: np.ndarray, d: np.ndarray, C: float, d_c: float, gamma: float, alpha: float) -> np.ndarray:
    """Critical: Ratio = C * |d - d_c|^(-gamma) * N^alpha"""
    t = np.abs(d - d_c)
    t_safe = np.maximum(t, 0.01)
    return C * np.power(t_safe, -gamma) * np.power(N, alpha)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def compute_bayes_factor(aic1: float, aic2: float) -> float:
    """
    Compute approximate Bayes factor from AIC values.
    BF = exp((AIC2 - AIC1) / 2)
    """
    return np.exp((aic2 - aic1) / 2)


def compute_aic(n: int, k: int, rss: float) -> float:
    """
    Compute AIC (Akaike Information Criterion).

    Args:
        n: Number of data points
        k: Number of parameters
        rss: Residual sum of squares
    """
    if rss <= 0:
        rss = 1e-10
    return n * np.log(rss / n) + 2 * k


def compute_bic(n: int, k: int, rss: float) -> float:
    """
    Compute BIC (Bayesian Information Criterion).
    """
    if rss <= 0:
        rss = 1e-10
    return n * np.log(rss / n) + k * np.log(n)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.99
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    """
    np.random.seed(RANDOM_SEED)
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (lower, upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# =============================================================================
# FIT UTILITIES
# =============================================================================

def fit_power_law(N: np.ndarray, d: np.ndarray, ratios: np.ndarray) -> Dict:
    """
    Fit power law model: Ratio = C * N^alpha * d^beta

    Returns:
        Dict with 'params', 'r_squared', 'aic', 'residuals'
    """
    from scipy.optimize import curve_fit

    # Flatten inputs
    N_flat = N.flatten()
    d_flat = d.flatten()
    ratios_flat = ratios.flatten()

    # Filter valid data (d > 0, ratio > 0)
    mask = (d_flat > 0.01) & (ratios_flat > 0)
    N_valid = N_flat[mask]
    d_valid = d_flat[mask]
    ratios_valid = ratios_flat[mask]

    if len(ratios_valid) < 4:
        return {'params': {}, 'r_squared': 0, 'aic': np.inf, 'residuals': np.array([])}

    def model(Nd, C, alpha, beta):
        N, d = Nd
        return C * np.power(N, alpha) * np.power(d, beta)

    try:
        popt, pcov = curve_fit(
            model,
            (N_valid, d_valid),
            ratios_valid,
            p0=[1.0, 1.0, 1.0],
            bounds=([0.01, -5, -5], [1000, 5, 5]),
            maxfev=10000
        )

        predicted = model((N_valid, d_valid), *popt)
        residuals = ratios_valid - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ratios_valid - np.mean(ratios_valid)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        aic = compute_aic(len(ratios_valid), 3, ss_res)

        return {
            'params': {'C': popt[0], 'alpha': popt[1], 'beta': popt[2]},
            'r_squared': r_squared,
            'aic': aic,
            'residuals': residuals
        }
    except Exception as e:
        return {'params': {}, 'r_squared': 0, 'aic': np.inf, 'residuals': np.array([]), 'error': str(e)}


def fit_all_models(N: np.ndarray, d: np.ndarray, ratios: np.ndarray) -> Dict[str, Dict]:
    """
    Fit all scaling law models and compare.

    Returns:
        Dict mapping model name to fit results
    """
    from scipy.optimize import curve_fit

    results = {}

    # Flatten and filter
    N_flat = N.flatten()
    d_flat = d.flatten()
    ratios_flat = ratios.flatten()

    mask = (d_flat > 0.01) & (ratios_flat > 0) & (N_flat >= 1)
    N_v = N_flat[mask]
    d_v = d_flat[mask]
    r_v = ratios_flat[mask]

    if len(r_v) < 5:
        return results

    # Power law
    results['power'] = fit_power_law(N, d, ratios)

    # Exponential
    def exp_model(Nd, C, alpha):
        N, d = Nd
        return C * np.exp(np.clip(alpha * N * d, -50, 50))

    try:
        popt, _ = curve_fit(exp_model, (N_v, d_v), r_v, p0=[1.0, 0.1],
                           bounds=([0.01, -2], [100, 2]), maxfev=5000)
        pred = exp_model((N_v, d_v), *popt)
        ss_res = np.sum((r_v - pred) ** 2)
        ss_tot = np.sum((r_v - np.mean(r_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['exponential'] = {
            'params': {'C': popt[0], 'alpha': popt[1]},
            'r_squared': r2,
            'aic': compute_aic(len(r_v), 2, ss_res)
        }
    except:
        results['exponential'] = {'params': {}, 'r_squared': 0, 'aic': np.inf}

    # Logarithmic
    def log_model(Nd, C, alpha, beta):
        N, d = Nd
        return C * (1 + alpha * np.log(np.maximum(N, 1))) * np.power(d, beta)

    try:
        popt, _ = curve_fit(log_model, (N_v, d_v), r_v, p0=[1.0, 1.0, 1.0],
                           bounds=([0.01, -10, -5], [100, 10, 5]), maxfev=5000)
        pred = log_model((N_v, d_v), *popt)
        ss_res = np.sum((r_v - pred) ** 2)
        ss_tot = np.sum((r_v - np.mean(r_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['logarithmic'] = {
            'params': {'C': popt[0], 'alpha': popt[1], 'beta': popt[2]},
            'r_squared': r2,
            'aic': compute_aic(len(r_v), 3, ss_res)
        }
    except:
        results['logarithmic'] = {'params': {}, 'r_squared': 0, 'aic': np.inf}

    # Linear
    def lin_model(Nd, C, alpha):
        N, d = Nd
        return C * (1 + alpha * N) * d

    try:
        popt, _ = curve_fit(lin_model, (N_v, d_v), r_v, p0=[1.0, 0.5],
                           bounds=([0.01, -10], [100, 10]), maxfev=5000)
        pred = lin_model((N_v, d_v), *popt)
        ss_res = np.sum((r_v - pred) ** 2)
        ss_tot = np.sum((r_v - np.mean(r_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['linear'] = {
            'params': {'C': popt[0], 'alpha': popt[1]},
            'r_squared': r2,
            'aic': compute_aic(len(r_v), 2, ss_res)
        }
    except:
        results['linear'] = {'params': {}, 'r_squared': 0, 'aic': np.inf}

    return results


# =============================================================================
# OUTPUT UTILITIES
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print()
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_result(label: str, value: Any, passed: Optional[bool] = None):
    """Print a labeled result."""
    status = ""
    if passed is not None:
        status = " [PASS]" if passed else " [FAIL]"
    print(f"  {label}: {value}{status}")


def print_metric(name: str, value: float, threshold: float, higher_is_better: bool = True):
    """Print a metric with pass/fail status."""
    if higher_is_better:
        passed = value >= threshold
    else:
        passed = value <= threshold
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {value:.4f} (threshold: {threshold:.4f}) [{status}]")


def save_results(results: Dict[str, ScalingLawResult], filepath: str):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'question': 'Q13: The 36x Ratio Scaling Law',
        'results': {k: v.to_dict() for k, v in results.items()}
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)


def print_summary(results: Dict[str, ScalingLawResult]):
    """Print summary of all test results."""
    print_header("Q13 TEST SUMMARY")

    passed_count = sum(1 for r in results.values() if r.passed)
    total = len(results)

    print()
    for test_id, result in sorted(results.items()):
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_name}")
        if result.scaling_exponents:
            exps = ", ".join(f"{k}={v:.3f}" for k, v in result.scaling_exponents.items())
            print(f"         Exponents: {exps}")
        if result.fit_quality > 0:
            print(f"         R^2: {result.fit_quality:.4f}")

    print()
    print("-" * 70)
    print(f"  Total: {passed_count}/{total} tests passed")
    print(f"  Threshold: {PASS_THRESHOLD}/{total} required")
    print()

    if passed_count >= PASS_THRESHOLD:
        print("  STATUS: Q13 ANSWERED")
    elif passed_count >= 7:
        print("  STATUS: Q13 PARTIAL")
    else:
        print("  STATUS: Q13 FALSIFIED or INCONCLUSIVE")
