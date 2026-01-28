"""
Q52 Chaos Theory Test - R's Relationship with Chaotic Dynamics

PRE-REGISTRATION:
    1. HYPOTHESIS: R inversely correlated with Lyapunov exponent (r < -0.5)
    2. PREDICTION: R detects edge of chaos, predicts bifurcations
    3. FALSIFICATION: If no correlation (|r| < 0.3)
    4. DATA: Logistic map x_{n+1} = r*x_n*(1-x_n) for r in [2.5, 4.0]
            Henon attractor (x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n)
    5. THRESHOLD: Characterize R's chaos detection capability

KEY INSIGHT:
    R (participation ratio / effective dimensionality) measures the "spread" of
    variance across dimensions. In chaotic dynamics:
    - Low Lyapunov exponent (regular/periodic): trajectory stays on low-dim manifold -> high R
    - High Lyapunov exponent (chaotic): trajectory fills phase space -> lower R
    - Edge of chaos: R should show transition behavior

LOGISTIC MAP GROUND TRUTH:
    - r < 3.0: stable fixed point (Lyapunov < 0)
    - r = 3.0: first bifurcation
    - r ~ 3.57: onset of chaos (Lyapunov = 0)
    - r > 3.57: chaos with periodic windows (Lyapunov > 0)
    - r = 4.0: fully chaotic (Lyapunov = ln(2))

TEST METHODOLOGY:
    1. Generate logistic map trajectories for r in [2.5, 4.0]
    2. Embed trajectories in higher dimension via delay embedding
    3. Compute Lyapunov exponent (known formula for logistic map)
    4. Compute R (participation ratio) on delay-embedded trajectories
    5. Correlate R with Lyapunov exponent
    6. Test bifurcation detection: does R change at known bifurcation points?
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import json
from scipy import stats

# Add paths
SCRIPT_DIR = Path(__file__).parent
QGT_PATH = SCRIPT_DIR.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
HARNESS_PATH = SCRIPT_DIR.parent / "q51"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(QGT_PATH))
sys.path.insert(0, str(HARNESS_PATH))

# Import from harness
try:
    from q51_test_harness import (
        BootstrapCI,
        NegativeControlResult,
        bootstrap_ci,
        cohens_d,
        compute_result_hash,
        format_ci,
        get_test_metadata,
        Q51Logger,
        Q51Seeds,
        Q51Thresholds,
    )
    HAS_HARNESS = True
except ImportError:
    HAS_HARNESS = False
    print("Warning: Q51 test harness not available, using minimal fallback")

# Import QGT for participation ratio
try:
    from qgt import participation_ratio, fubini_study_metric
    HAS_QGT = True
except ImportError:
    HAS_QGT = False
    print("Warning: QGT library not available, using inline implementation")


# =============================================================================
# CONSTANTS
# =============================================================================

# Logistic map parameters
R_MIN = 2.5      # Start of parameter sweep
R_MAX = 4.0      # End of parameter sweep
N_R_VALUES = 100 # Number of r values to test

# Known bifurcation points for logistic map
BIFURCATION_POINTS = {
    'first': 3.0,           # First bifurcation (period-1 to period-2)
    'second': 3.449,        # Period-2 to period-4
    'onset_chaos': 3.5699,  # Onset of chaos (accumulation point)
    'fully_chaotic': 4.0    # Fully chaotic
}

# Lyapunov exponent thresholds
LYAPUNOV_CHAOS_THRESHOLD = 0.0  # Positive = chaos, negative = regular

# Trajectory parameters
TRAJECTORY_LENGTH = 10000    # Length of each trajectory
TRANSIENT_LENGTH = 1000      # Initial transient to discard
DELAY_EMBEDDING_DIM = 3      # Dimension for delay embedding
DELAY_TAU = 1                # Delay for embedding

# Correlation thresholds (from pre-registration)
CORRELATION_STRONG_NEGATIVE = -0.5   # Hypothesis threshold
CORRELATION_WEAK = 0.3               # Falsification threshold

# Seeds for reproducibility
SEED_LOGISTIC = 42
SEED_HENON = 43
SEED_BOOTSTRAP = 9999


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LogisticMapPoint:
    """Result for a single r value in the logistic map."""
    r_param: float
    lyapunov_exponent: float
    participation_ratio: float
    trajectory_length: int
    is_chaotic: bool
    regime: str  # 'fixed', 'periodic', 'edge', 'chaotic'


@dataclass
class ChaosCorrelationResult:
    """Result of R vs Lyapunov correlation analysis."""
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n_points: int
    hypothesis_supported: bool
    verdict: str


@dataclass
class BifurcationDetectionResult:
    """Result of bifurcation detection test."""
    bifurcation_name: str
    r_value: float
    r_gradient_at_bifurcation: float
    detected: bool
    detection_threshold: float


@dataclass
class Q52Result:
    """Complete Q52 test result."""
    test_name: str
    hypothesis: str
    prediction: str
    correlation_result: Dict
    bifurcation_results: List[Dict]
    negative_control: Dict
    henon_result: Optional[Dict]
    overall_status: str
    verdict: str
    metadata: Dict


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def logistic_map_trajectory(
    r: float,
    n_points: int,
    x0: float = 0.5,
    transient: int = TRANSIENT_LENGTH
) -> np.ndarray:
    """
    Generate trajectory from logistic map: x_{n+1} = r * x_n * (1 - x_n)

    Args:
        r: Control parameter (bifurcation parameter)
        n_points: Number of points after transient
        x0: Initial condition
        transient: Number of initial points to discard

    Returns:
        1D array of trajectory points
    """
    total_points = n_points + transient
    trajectory = np.zeros(total_points)
    trajectory[0] = x0

    for i in range(1, total_points):
        x = trajectory[i-1]
        trajectory[i] = r * x * (1 - x)

        # Numerical stability: clip to [0, 1]
        trajectory[i] = np.clip(trajectory[i], 1e-10, 1 - 1e-10)

    # Discard transient
    return trajectory[transient:]


def compute_lyapunov_logistic(r: float, n_iterations: int = 10000) -> float:
    """
    Compute Lyapunov exponent for logistic map.

    For logistic map f(x) = r*x*(1-x):
        f'(x) = r*(1 - 2x)
        Lyapunov = lim_{n->inf} (1/n) * sum_{i=0}^{n-1} ln|f'(x_i)|
                 = lim_{n->inf} (1/n) * sum_{i=0}^{n-1} ln|r*(1 - 2*x_i)|

    For r = 4: Lyapunov = ln(2) ~ 0.693 (theoretically known)
    """
    x = 0.5  # Initial condition
    lyapunov_sum = 0.0
    transient = 1000

    # Transient
    for _ in range(transient):
        x = r * x * (1 - x)
        x = np.clip(x, 1e-10, 1 - 1e-10)

    # Compute Lyapunov
    for _ in range(n_iterations):
        # Derivative: f'(x) = r * (1 - 2x)
        derivative = r * (1 - 2 * x)
        if abs(derivative) > 1e-10:
            lyapunov_sum += np.log(abs(derivative))

        x = r * x * (1 - x)
        x = np.clip(x, 1e-10, 1 - 1e-10)

    return lyapunov_sum / n_iterations


def delay_embed(trajectory: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """
    Create delay embedding of a 1D trajectory.

    Takens' embedding theorem: a scalar time series can be embedded in
    d-dimensional space using time-delayed copies.

    X_i = [x(i), x(i+tau), x(i+2*tau), ..., x(i+(dim-1)*tau)]

    Args:
        trajectory: 1D time series
        dim: Embedding dimension
        tau: Time delay

    Returns:
        (n_embedded, dim) array
    """
    n = len(trajectory)
    n_embedded = n - (dim - 1) * tau

    if n_embedded <= 0:
        raise ValueError(f"Trajectory too short for embedding: {n} < {(dim-1)*tau + 1}")

    embedded = np.zeros((n_embedded, dim))
    for d in range(dim):
        embedded[:, d] = trajectory[d * tau : d * tau + n_embedded]

    return embedded


def compute_participation_ratio_inline(embeddings: np.ndarray) -> float:
    """
    Compute participation ratio (effective dimensionality).

    Df = (sum(lambda))^2 / sum(lambda^2)

    This is the R metric we're testing against chaos.
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    if sum_lambda_sq < 1e-20:
        return 0.0

    return (sum_lambda ** 2) / sum_lambda_sq


def compute_R(embeddings: np.ndarray) -> float:
    """Compute R (participation ratio) - wrapper for available implementation."""
    if HAS_QGT:
        return participation_ratio(embeddings, normalize=False)
    else:
        return compute_participation_ratio_inline(embeddings)


def classify_regime(r: float, lyapunov: float) -> str:
    """Classify dynamical regime based on r and Lyapunov exponent."""
    if r < 3.0:
        return 'fixed'
    elif r < 3.449:
        return 'period_2'
    elif r < BIFURCATION_POINTS['onset_chaos']:
        return 'periodic'
    elif lyapunov > 0.1:
        return 'chaotic'
    else:
        return 'edge'


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_logistic_map_sweep(
    r_values: np.ndarray,
    verbose: bool = True
) -> Tuple[List[LogisticMapPoint], ChaosCorrelationResult]:
    """
    Sweep through r values and compute R vs Lyapunov correlation.

    This is the main test of the hypothesis.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("LOGISTIC MAP SWEEP: R vs LYAPUNOV EXPONENT")
        print("=" * 60)

    results = []
    lyapunov_values = []
    R_values = []

    for i, r in enumerate(r_values):
        # Generate trajectory
        trajectory = logistic_map_trajectory(r, TRAJECTORY_LENGTH)

        # Delay embedding
        embedded = delay_embed(trajectory, DELAY_EMBEDDING_DIM, DELAY_TAU)

        # Compute metrics
        lyapunov = compute_lyapunov_logistic(r)
        R = compute_R(embedded)

        # Classify regime
        regime = classify_regime(r, lyapunov)
        is_chaotic = lyapunov > LYAPUNOV_CHAOS_THRESHOLD

        result = LogisticMapPoint(
            r_param=float(r),
            lyapunov_exponent=float(lyapunov),
            participation_ratio=float(R),
            trajectory_length=len(trajectory),
            is_chaotic=is_chaotic,
            regime=regime
        )
        results.append(result)

        lyapunov_values.append(lyapunov)
        R_values.append(R)

        if verbose and (i % 20 == 0 or i == len(r_values) - 1):
            print(f"  r={r:.3f}: Lyapunov={lyapunov:+.4f}, R={R:.3f}, regime={regime}")

    # Compute correlations
    lyapunov_arr = np.array(lyapunov_values)
    R_arr = np.array(R_values)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(lyapunov_arr, R_arr)

    # Spearman correlation (more robust to outliers)
    spearman_rho, spearman_p = stats.spearmanr(lyapunov_arr, R_arr)

    # Hypothesis test
    # H0: R is inversely correlated with Lyapunov (r < -0.5)
    # H1 (falsification): No correlation (|r| < 0.3)
    if pearson_r < CORRELATION_STRONG_NEGATIVE:
        hypothesis_supported = True
        verdict = f"CONFIRMED: Strong negative correlation (r={pearson_r:.3f} < {CORRELATION_STRONG_NEGATIVE})"
    elif abs(pearson_r) < CORRELATION_WEAK:
        hypothesis_supported = False
        verdict = f"FALSIFIED: No significant correlation (|r|={abs(pearson_r):.3f} < {CORRELATION_WEAK})"
    else:
        hypothesis_supported = pearson_r < 0
        if pearson_r < 0:
            verdict = f"PARTIAL: Weak negative correlation (r={pearson_r:.3f})"
        else:
            verdict = f"UNEXPECTED: Positive correlation (r={pearson_r:.3f})"

    correlation_result = ChaosCorrelationResult(
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        n_points=len(r_values),
        hypothesis_supported=hypothesis_supported,
        verdict=verdict
    )

    if verbose:
        print("\n" + "-" * 40)
        print("CORRELATION RESULTS:")
        print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.4e})")
        print(f"  Spearman rho = {spearman_rho:.4f} (p = {spearman_p:.4e})")
        print(f"  Verdict: {verdict}")

    return results, correlation_result


def test_bifurcation_detection(
    logistic_results: List[LogisticMapPoint],
    verbose: bool = True
) -> List[BifurcationDetectionResult]:
    """
    Test whether R changes detectably at known bifurcation points.

    At bifurcation, the dynamical regime changes, so R should show
    a transition (gradient peak or inflection point).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BIFURCATION DETECTION TEST")
        print("=" * 60)

    # Extract r and R values
    r_vals = np.array([p.r_param for p in logistic_results])
    R_vals = np.array([p.participation_ratio for p in logistic_results])

    # Compute gradient of R with respect to r
    dr = np.diff(r_vals)
    dR = np.diff(R_vals)
    gradient = dR / dr
    gradient_r = (r_vals[:-1] + r_vals[1:]) / 2  # Midpoints

    # Standard deviation of gradient for threshold
    gradient_std = np.std(gradient)
    detection_threshold = 2.0 * gradient_std  # 2 sigma threshold

    results = []
    for name, r_bif in BIFURCATION_POINTS.items():
        # Find closest gradient point
        idx = np.argmin(np.abs(gradient_r - r_bif))
        local_gradient = abs(gradient[idx])

        # Check if gradient exceeds threshold
        detected = local_gradient > detection_threshold

        result = BifurcationDetectionResult(
            bifurcation_name=name,
            r_value=float(r_bif),
            r_gradient_at_bifurcation=float(local_gradient),
            detected=detected,
            detection_threshold=float(detection_threshold)
        )
        results.append(result)

        if verbose:
            status = "DETECTED" if detected else "NOT DETECTED"
            print(f"  {name} (r={r_bif:.3f}): |dR/dr|={local_gradient:.4f} [{status}]")

    return results


def test_henon_attractor(verbose: bool = True) -> Dict:
    """
    Test R on Henon attractor as additional chaotic system.

    Henon map:
        x_{n+1} = 1 - a * x_n^2 + y_n
        y_{n+1} = b * x_n

    Standard parameters: a = 1.4, b = 0.3 (chaotic)
    Lyapunov exponents: lambda_1 ~ 0.42, lambda_2 ~ -1.62
    """
    if verbose:
        print("\n" + "=" * 60)
        print("HENON ATTRACTOR TEST")
        print("=" * 60)

    # Standard chaotic parameters
    a_chaotic = 1.4
    b_chaotic = 0.3

    # Non-chaotic parameters (smaller a)
    a_regular = 0.2
    b_regular = 0.3

    np.random.seed(SEED_HENON)

    def henon_trajectory(a, b, n_points, transient=1000):
        """Generate Henon map trajectory."""
        total = n_points + transient
        x, y = np.zeros(total), np.zeros(total)
        x[0], y[0] = 0.1, 0.1

        for i in range(1, total):
            x[i] = 1 - a * x[i-1]**2 + y[i-1]
            y[i] = b * x[i-1]

        return np.column_stack([x[transient:], y[transient:]])

    def henon_lyapunov(a, b, n_iter=10000):
        """Estimate largest Lyapunov exponent for Henon map."""
        x, y = 0.1, 0.1
        lyapunov_sum = 0.0

        # Small perturbation
        dx, dy = 1e-8, 1e-8

        for _ in range(1000):  # Transient
            x_new = 1 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new

        for _ in range(n_iter):
            # Jacobian
            J = np.array([[-2*a*x, 1], [b, 0]])

            # Perturbed trajectory
            perturbation = np.array([dx, dy])
            perturbation = J @ perturbation

            # Renormalize
            norm = np.linalg.norm(perturbation)
            if norm > 1e-10:
                lyapunov_sum += np.log(norm)
                perturbation = perturbation / norm

            dx, dy = perturbation

            # Iterate base trajectory
            x_new = 1 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new

        return lyapunov_sum / n_iter

    # Generate trajectories
    traj_chaotic = henon_trajectory(a_chaotic, b_chaotic, 5000)
    traj_regular = henon_trajectory(a_regular, b_regular, 5000)

    # Compute metrics
    R_chaotic = compute_R(traj_chaotic)
    R_regular = compute_R(traj_regular)
    lyapunov_chaotic = henon_lyapunov(a_chaotic, b_chaotic)
    lyapunov_regular = henon_lyapunov(a_regular, b_regular)

    if verbose:
        print(f"  Chaotic (a={a_chaotic}): Lyapunov={lyapunov_chaotic:.4f}, R={R_chaotic:.4f}")
        print(f"  Regular (a={a_regular}): Lyapunov={lyapunov_regular:.4f}, R={R_regular:.4f}")

    # Test: R should be different for chaotic vs regular
    R_difference = R_regular - R_chaotic  # Expect positive if hypothesis holds
    consistent_with_hypothesis = (lyapunov_chaotic > lyapunov_regular) and (R_chaotic < R_regular)

    result = {
        'chaotic': {
            'a': a_chaotic, 'b': b_chaotic,
            'lyapunov': float(lyapunov_chaotic),
            'R': float(R_chaotic)
        },
        'regular': {
            'a': a_regular, 'b': b_regular,
            'lyapunov': float(lyapunov_regular),
            'R': float(R_regular)
        },
        'R_difference': float(R_difference),
        'consistent_with_hypothesis': consistent_with_hypothesis,
        'verdict': "CONSISTENT" if consistent_with_hypothesis else "INCONSISTENT"
    }

    if verbose:
        print(f"  R difference (regular - chaotic): {R_difference:.4f}")
        print(f"  Verdict: {result['verdict']}")

    return result


def run_negative_control(verbose: bool = True) -> NegativeControlResult:
    """
    Negative control: Random noise should show no R-Lyapunov relationship.

    We generate random white noise and compute R. There should be no
    meaningful structure, so R should be constant regardless of any
    "artificial Lyapunov" computed from random data.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("NEGATIVE CONTROL: Random Noise")
        print("=" * 60)

    np.random.seed(SEED_BOOTSTRAP)

    # Generate multiple random trajectories and compute R
    n_trials = 20
    R_random = []

    for _ in range(n_trials):
        random_traj = np.random.randn(TRAJECTORY_LENGTH)
        embedded = delay_embed(random_traj, DELAY_EMBEDDING_DIM, DELAY_TAU)
        R = compute_R(embedded)
        R_random.append(R)

    R_random = np.array(R_random)
    cv = np.std(R_random) / np.mean(R_random)  # Coefficient of variation

    # Random noise should have consistent R (low CV)
    cv_threshold = 0.1  # CV < 10%
    test_passed = cv < cv_threshold

    if verbose:
        print(f"  Mean R (random): {np.mean(R_random):.4f}")
        print(f"  Std R (random): {np.std(R_random):.4f}")
        print(f"  CV: {cv:.4f} (threshold: < {cv_threshold})")
        print(f"  Status: {'PASS' if test_passed else 'FAIL'}")

    return NegativeControlResult(
        name="random_noise_consistent_R",
        test_passed=test_passed,
        expected_behavior=f"Random noise should have consistent R (CV < {cv_threshold})",
        actual_behavior=f"CV = {cv:.4f}",
        metric_value=float(cv),
        metric_threshold=cv_threshold,
        notes="Random white noise has no chaotic structure, R should be constant"
    )


# =============================================================================
# MAIN TEST
# =============================================================================

def run_q52_chaos_test(verbose: bool = True) -> Q52Result:
    """Run the complete Q52 chaos theory test."""
    print("\n" + "=" * 70)
    print("Q52 CHAOS THEORY TEST: R's RELATIONSHIP WITH CHAOTIC DYNAMICS")
    print("=" * 70)
    print("\nHypothesis: R inversely correlated with Lyapunov exponent (r < -0.5)")
    print("Prediction: R detects edge of chaos, predicts bifurcations")
    print("Falsification: No correlation (|r| < 0.3)")

    # 1. Logistic map sweep
    r_values = np.linspace(R_MIN, R_MAX, N_R_VALUES)
    logistic_results, correlation_result = test_logistic_map_sweep(r_values, verbose)

    # 2. Bifurcation detection
    bifurcation_results = test_bifurcation_detection(logistic_results, verbose)

    # 3. Henon attractor
    henon_result = test_henon_attractor(verbose)

    # 4. Negative control
    negative_control = run_negative_control(verbose)

    # Overall assessment
    bifurcations_detected = sum(1 for b in bifurcation_results if b.detected)
    total_bifurcations = len(bifurcation_results)

    # Determine overall status
    if correlation_result.hypothesis_supported and negative_control.test_passed:
        overall_status = "CONFIRMED"
    elif not negative_control.test_passed:
        overall_status = "INCONCLUSIVE (negative control failed)"
    elif correlation_result.hypothesis_supported:
        overall_status = "PARTIAL"
    else:
        overall_status = "FALSIFIED"

    # Construct verdict
    verdict = (
        f"R-Lyapunov correlation: {correlation_result.verdict}\n"
        f"Bifurcations detected: {bifurcations_detected}/{total_bifurcations}\n"
        f"Henon test: {henon_result['verdict']}\n"
        f"Negative control: {'PASS' if negative_control.test_passed else 'FAIL'}"
    )

    # Get metadata
    metadata = {}
    if HAS_HARNESS:
        metadata = get_test_metadata()
    else:
        metadata = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'numpy_version': np.__version__
        }

    result = Q52Result(
        test_name="Q52_CHAOS_THEORY",
        hypothesis="R inversely correlated with Lyapunov exponent (r < -0.5)",
        prediction="R detects edge of chaos, predicts bifurcations",
        correlation_result=asdict(correlation_result),
        bifurcation_results=[asdict(b) for b in bifurcation_results],
        negative_control=negative_control.to_dict() if HAS_HARNESS else {
            'name': negative_control.name,
            'test_passed': negative_control.test_passed,
            'metric_value': negative_control.metric_value
        },
        henon_result=henon_result,
        overall_status=overall_status,
        verdict=verdict,
        metadata=metadata
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Q52 SUMMARY")
    print("=" * 70)
    print(f"\nPearson correlation (R vs Lyapunov): {correlation_result.pearson_r:.4f}")
    print(f"Spearman correlation (R vs Lyapunov): {correlation_result.spearman_rho:.4f}")
    print(f"Bifurcations detected: {bifurcations_detected}/{total_bifurcations}")
    print(f"\nOVERALL STATUS: {overall_status}")
    print("-" * 70)
    print(verdict)
    print("=" * 70)

    return result


def save_results(result: Q52Result, output_dir: Path = None):
    """Save results to JSON file."""
    if output_dir is None:
        output_dir = SCRIPT_DIR / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    output = asdict(result)

    # Compute hash
    if HAS_HARNESS:
        output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q52_chaos_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = run_q52_chaos_test(verbose=True)
    save_results(result)

    # Exit code based on hypothesis support
    if "CONFIRMED" in result.overall_status:
        sys.exit(0)
    elif "FALSIFIED" in result.overall_status:
        sys.exit(1)
    else:
        sys.exit(2)  # Partial or inconclusive
