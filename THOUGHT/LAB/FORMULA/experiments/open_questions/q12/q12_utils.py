"""
Q12: Phase Transitions in Semantic Systems - Utilities

Shared dataclasses, enums, and utility functions for the 12 HARDCORE tests.

These tests apply statistical physics methods to determine whether
semantic systems exhibit true phase transitions where "truth crystallizes
suddenly rather than gradually."

Author: AGS Research
Date: 2026-01-19
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# ENUMS
# =============================================================================

class TransitionType(Enum):
    """Classification of phase transition order."""
    FIRST_ORDER = "first_order"       # Discontinuous jump (latent heat)
    SECOND_ORDER = "second_order"     # Continuous but singular derivatives
    INFINITE_ORDER = "infinite_order" # Kosterlitz-Thouless type
    CROSSOVER = "crossover"           # Smooth (no true transition)
    UNKNOWN = "unknown"


class UniversalityClass(Enum):
    """Known universality classes from statistical physics."""
    ISING_2D = "ising_2d"             # nu=1.0, beta=0.125, gamma=1.75
    ISING_3D = "ising_3d"             # nu=0.63, beta=0.326, gamma=1.24
    PERCOLATION_2D = "percolation_2d" # nu=1.33, beta=0.14, gamma=2.39
    PERCOLATION_3D = "percolation_3d" # nu=0.88, beta=0.41, gamma=1.80
    MEAN_FIELD = "mean_field"         # nu=0.5, beta=0.5, gamma=1.0
    XY_2D = "xy_2d"                   # Kosterlitz-Thouless
    HEISENBERG_3D = "heisenberg_3d"   # nu=0.71, beta=0.37, gamma=1.39
    UNKNOWN = "unknown"


class TestResult(Enum):
    """Result of a single test."""
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


# =============================================================================
# UNIVERSALITY CLASS REFERENCE DATA
# =============================================================================

UNIVERSALITY_EXPONENTS = {
    UniversalityClass.ISING_2D: {"nu": 1.0, "beta": 0.125, "gamma": 1.75, "alpha": 0.0},
    UniversalityClass.ISING_3D: {"nu": 0.6301, "beta": 0.3265, "gamma": 1.2372, "alpha": 0.110},
    UniversalityClass.PERCOLATION_2D: {"nu": 1.333, "beta": 0.1389, "gamma": 2.389, "alpha": -0.667},
    UniversalityClass.PERCOLATION_3D: {"nu": 0.8765, "beta": 0.4181, "gamma": 1.7933, "alpha": -0.625},
    UniversalityClass.MEAN_FIELD: {"nu": 0.5, "beta": 0.5, "gamma": 1.0, "alpha": 0.0},
    UniversalityClass.HEISENBERG_3D: {"nu": 0.7112, "beta": 0.3689, "gamma": 1.3960, "alpha": -0.133},
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CriticalExponents:
    """Critical exponents characterizing a phase transition."""
    nu: float = 0.0       # Correlation length: xi ~ |t|^(-nu)
    beta: float = 0.0     # Order parameter: M ~ |t|^beta
    gamma: float = 0.0    # Susceptibility: chi ~ |t|^(-gamma)
    alpha: float = 0.0    # Specific heat: C ~ |t|^(-alpha)
    delta: float = 0.0    # Equation of state: M ~ H^(1/delta) at t=0
    eta: float = 0.0      # Anomalous dimension: G(r) ~ r^(-(d-2+eta))
    z: float = 0.0        # Dynamic exponent: tau ~ xi^z

    def hyperscaling_check(self, d: int = 3) -> float:
        """Check hyperscaling relation: 2 - alpha = d * nu"""
        lhs = 2 - self.alpha
        rhs = d * self.nu
        return abs(lhs - rhs)

    def scaling_check(self) -> float:
        """Check Rushbrooke scaling: alpha + 2*beta + gamma = 2"""
        return abs(self.alpha + 2 * self.beta + self.gamma - 2)

    def distance_to_class(self, uc: UniversalityClass) -> float:
        """Euclidean distance to a known universality class."""
        if uc not in UNIVERSALITY_EXPONENTS:
            return float('inf')
        ref = UNIVERSALITY_EXPONENTS[uc]
        return np.sqrt(
            (self.nu - ref["nu"])**2 +
            (self.beta - ref["beta"])**2 +
            (self.gamma - ref["gamma"])**2
        )

    def nearest_class(self) -> Tuple[UniversalityClass, float]:
        """Find nearest universality class."""
        best_class = UniversalityClass.UNKNOWN
        best_dist = float('inf')
        for uc in UNIVERSALITY_EXPONENTS:
            d = self.distance_to_class(uc)
            if d < best_dist:
                best_dist = d
                best_class = uc
        return best_class, best_dist


@dataclass
class PhaseTransitionTestResult:
    """Result from a single phase transition test."""
    test_name: str
    test_id: str
    passed: bool
    metric_value: float
    threshold: float
    transition_type: TransitionType = TransitionType.UNKNOWN
    universality_class: UniversalityClass = UniversalityClass.UNKNOWN
    critical_point: Optional[float] = None
    critical_exponents: CriticalExponents = field(default_factory=CriticalExponents)
    evidence: Dict[str, Any] = field(default_factory=dict)
    falsification_evidence: Optional[str] = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["transition_type"] = self.transition_type.value
        d["universality_class"] = self.universality_class.value
        return d


@dataclass
class TestConfig:
    """Configuration for test runs."""
    seed: int = 42
    n_trials: int = 100
    n_samples: int = 1000
    alpha_resolution: int = 50  # Number of alpha points between 0 and 1
    system_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    verbose: bool = True


# =============================================================================
# THRESHOLDS - Pre-specified for rigor
# =============================================================================

THRESHOLDS = {
    # Test 1: Finite-Size Scaling
    "collapse_r2": 0.90,
    "nu_consistency_cv": 0.15,

    # Test 2: Universality Class
    "class_distance": 0.25,
    "hyperscaling_violation": 0.20,

    # Test 3: Susceptibility Divergence
    "susceptibility_ratio": 50.0,
    "susceptibility_fwhm": 0.10,

    # Test 4: Critical Slowing Down
    "tau_ratio": 10.0,

    # Test 5: Hysteresis
    "hysteresis_area_first_order": 0.05,
    "hysteresis_area_second_order": 0.02,

    # Test 6: Order Parameter Jump
    "jump_ratio": 2.0,

    # Test 7: Percolation
    "giant_component_high": 0.80,
    "giant_component_low": 0.20,
    "transition_sharpness": 0.15,

    # Test 8: Scale Invariance
    "power_law_r2": 0.92,
    "exponential_r2": 0.90,

    # Test 9: Binder Cumulant
    "crossing_spread": 0.03,
    "binder_value_min": 0.4,
    "binder_value_max": 0.7,

    # Test 10: Fisher Information
    "fisher_ratio": 20.0,
    "fisher_fwhm": 0.15,

    # Test 11: Symmetry Breaking
    "isotropy_ratio": 3.0,

    # Test 12: Cross-Architecture
    "alpha_c_cv": 0.20,
    "exponent_agreement": 0.15,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def compute_entropy(data: np.ndarray, bins: int = 30) -> float:
    """Compute Shannon entropy of data using histogram binning."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_vars = data.shape[1]

    if n_vars == 1:
        hist, _ = np.histogram(data, bins=bins, density=False)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        return -np.sum(prob * np.log(prob))
    else:
        # Multidimensional - sum of marginals as approximation
        return sum(compute_entropy(data[:, i:i+1], bins) for i in range(n_vars))


def fit_power_law(x: np.ndarray, y: np.ndarray,
                  x0: float = 0.0) -> Tuple[float, float, float]:
    """
    Fit y ~ |x - x0|^exponent

    Returns: (exponent, amplitude, r_squared)
    """
    # Remove points too close to x0
    mask = np.abs(x - x0) > 0.01
    if np.sum(mask) < 5:
        return 0.0, 0.0, 0.0

    x_fit = np.abs(x[mask] - x0)
    y_fit = y[mask]

    # Avoid log of zero
    valid = (x_fit > 0) & (y_fit > 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0, 0.0

    log_x = np.log(x_fit[valid])
    log_y = np.log(y_fit[valid])

    # Linear fit in log-log space
    coeffs = np.polyfit(log_x, log_y, 1)
    exponent = coeffs[0]
    amplitude = np.exp(coeffs[1])

    # R-squared
    y_pred = amplitude * x_fit[valid] ** exponent
    ss_res = np.sum((y_fit[valid] - y_pred) ** 2)
    ss_tot = np.sum((y_fit[valid] - np.mean(y_fit[valid])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return exponent, amplitude, r2


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y ~ A * exp(-x / xi)

    Returns: (xi, amplitude, r_squared)
    """
    valid = y > 0
    if np.sum(valid) < 5:
        return 0.0, 0.0, 0.0

    log_y = np.log(y[valid])
    x_fit = x[valid]

    # Linear fit of log(y) vs x
    coeffs = np.polyfit(x_fit, log_y, 1)
    slope = coeffs[0]
    xi = -1.0 / slope if slope < 0 else float('inf')
    amplitude = np.exp(coeffs[1])

    # R-squared
    y_pred = amplitude * np.exp(-x_fit / xi) if xi != float('inf') else amplitude * np.ones_like(x_fit)
    ss_res = np.sum((y[valid] - y_pred) ** 2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return xi, amplitude, r2


def find_crossing_point(x: np.ndarray, curves: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Find the crossing point where multiple curves intersect.

    Returns: (x_crossing, spread, y_value_at_crossing)
    """
    n_curves = len(curves)
    if n_curves < 2:
        return 0.0, float('inf'), 0.0

    # Find approximate crossing region by minimizing variance
    variances = []
    for i, xi in enumerate(x):
        values = [c[i] for c in curves if i < len(c)]
        variances.append(np.var(values))

    min_idx = np.argmin(variances)
    x_crossing = x[min_idx]
    spread = np.sqrt(variances[min_idx])
    y_value = np.mean([c[min_idx] for c in curves if min_idx < len(c)])

    return x_crossing, spread, y_value


def compute_binder_cumulant(order_parameter_samples: np.ndarray) -> float:
    """
    Compute Binder cumulant U = 1 - <M^4> / (3 * <M^2>^2)

    At critical point: U is system-size independent
    Below T_c: U -> 2/3 (ordered phase)
    Above T_c: U -> 0 (disordered phase)
    """
    m2 = np.mean(order_parameter_samples ** 2)
    m4 = np.mean(order_parameter_samples ** 4)

    if m2 < 1e-10:
        return 0.0

    return 1.0 - m4 / (3.0 * m2 ** 2)


def compute_susceptibility(order_parameter_samples: np.ndarray,
                           system_size: int) -> float:
    """
    Compute magnetic susceptibility chi = N * (<M^2> - <M>^2)
    """
    m_mean = np.mean(order_parameter_samples)
    m2_mean = np.mean(order_parameter_samples ** 2)

    return system_size * (m2_mean - m_mean ** 2)


def compute_correlation_function(data: np.ndarray,
                                  max_r: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation function C(r) from spatial data.

    Returns: (r_values, correlation_values)
    """
    n = len(data)
    max_r = min(max_r, n // 2)

    # Normalize data
    data_centered = data - np.mean(data)
    variance = np.var(data)
    if variance < 1e-10:
        return np.arange(max_r), np.zeros(max_r)

    # Compute correlation at each distance
    r_values = np.arange(max_r)
    correlations = np.zeros(max_r)

    for r in range(max_r):
        if r == 0:
            correlations[r] = 1.0
        else:
            correlations[r] = np.mean(data_centered[:-r] * data_centered[r:]) / variance

    return r_values, correlations


def compute_hysteresis_area(forward: np.ndarray, reverse: np.ndarray,
                            x: np.ndarray) -> float:
    """
    Compute area between forward and reverse paths.
    """
    return np.trapz(np.abs(forward - reverse), x)


def compute_fisher_information(log_prob_gradient: np.ndarray) -> float:
    """
    Compute Fisher information I = E[(d log P / d alpha)^2]
    """
    return np.mean(log_prob_gradient ** 2)


def compute_isotropy(covariance: np.ndarray) -> float:
    """
    Compute isotropy as ratio of smallest to largest eigenvalue.

    Isotropy = 1: perfectly isotropic
    Isotropy = 0: completely anisotropic
    """
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove numerical zeros

    if len(eigenvalues) == 0:
        return 0.0

    return eigenvalues.min() / eigenvalues.max()


def find_largest_cluster(adjacency: np.ndarray) -> int:
    """
    Find size of largest connected component using BFS.
    """
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    largest = 0

    for start in range(n):
        if visited[start]:
            continue

        # BFS from this node
        queue = [start]
        visited[start] = True
        cluster_size = 0

        while queue:
            node = queue.pop(0)
            cluster_size += 1

            for neighbor in range(n):
                if adjacency[node, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        largest = max(largest, cluster_size)

    return largest


def save_results(results: Dict[str, PhaseTransitionTestResult],
                 filepath: str):
    """Save test results to JSON file."""
    output = {
        "test_suite": "Q12_Phase_Transitions",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results.values() if r.passed),
            "failed": sum(1 for r in results.values() if not r.passed),
        },
        "results": {k: v.to_dict() for k, v in results.items()}
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def print_summary(results: Dict[str, PhaseTransitionTestResult]):
    """Print summary of test results."""
    print("=" * 70)
    print("Q12: PHASE TRANSITIONS - TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)

    print(f"\nOverall: {passed}/{total} tests PASSED")
    print()

    for test_id, result in results.items():
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"  {status} {result.test_name}")
        print(f"         Metric: {result.metric_value:.4f} (threshold: {result.threshold})")
        if result.critical_point is not None:
            print(f"         Critical point: alpha_c = {result.critical_point:.4f}")
        if result.falsification_evidence:
            print(f"         Falsification: {result.falsification_evidence}")
        print()

    # Verdict
    print("=" * 70)
    if passed >= 10:
        print("VERDICT: ** ANSWERED ** - Phase transition CONFIRMED")
    elif passed >= 7:
        print("VERDICT: PARTIAL - Strong evidence, some tests inconclusive")
    else:
        print("VERDICT: FALSIFIED - Not a true phase transition")
    print("=" * 70)
