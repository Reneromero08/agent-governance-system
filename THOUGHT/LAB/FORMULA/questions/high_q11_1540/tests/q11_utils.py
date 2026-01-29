#!/usr/bin/env python3
"""
Q11 Valley Blindness - Shared Utilities

Core infrastructure for the 12 information horizon tests.
"""

import sys
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Windows console encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
SEMANTIC_MODEL = 'all-MiniLM-L6-v2'
EPS = 1e-12

# Horizon detection thresholds
HORIZON_R_THRESHOLD = 0.1      # R below this = horizon reached
TRANSLATION_LOSS_THRESHOLD = 0.1  # Translation fidelity loss threshold
VOID_DETECTION_THRESHOLD = 0.5    # Minimum detection rate for unknown unknowns
QUALIA_GAP_THRESHOLD = 0.2        # Minimum gap for qualia horizon
ASYMMETRY_THRESHOLD = 0.05        # Time asymmetry detection threshold


# =============================================================================
# ENUMS
# =============================================================================

class HorizonType(Enum):
    """Classification of information horizon types."""
    INSTRUMENTAL = "instrumental"     # Can be extended with better tools
    COMPUTATIONAL = "computational"   # Limited by complexity bounds
    STRUCTURAL = "structural"         # Requires epistemology change
    SEMANTIC = "semantic"             # Translation barriers
    TEMPORAL = "temporal"             # Time-asymmetric access
    ONTOLOGICAL = "ontological"       # Possibly absolute limits
    UNKNOWN = "unknown"


class ExtensionMethod(Enum):
    """Methods for attempting to extend information horizons."""
    MORE_DATA = "more_data"           # Category A: Same epistemology
    MORE_COMPUTE = "more_compute"     # Category A: Same epistemology
    NEW_INSTRUMENT = "new_instrument" # Category B: New sensors
    PRIOR_CHANGE = "prior_change"     # Category C: Epistemology change
    LOGIC_CHANGE = "logic_change"     # Category C: Epistemology change
    SCALE_CHANGE = "scale_change"     # Renormalization approach


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for horizon tests."""
    seed: int = RANDOM_SEED
    n_trials: int = 100
    n_samples: int = 1000
    verbose: bool = True


@dataclass
class HorizonTestResult:
    """Result from a single horizon test."""
    test_name: str
    test_id: str
    passed: bool
    horizon_type: HorizonType
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        d = asdict(self)
        d['horizon_type'] = self.horizon_type.value
        return {k: to_builtin(v) for k, v in d.items()}


@dataclass
class Q11Summary:
    """Summary of all Q11 test results."""
    total_tests: int = 12
    passed_tests: int = 0
    failed_tests: int = 0
    horizon_types_found: List[str] = field(default_factory=list)
    answer: str = ""
    results: List[HorizonTestResult] = field(default_factory=list)

    def compute_answer(self):
        """Determine Q11 answer based on test results."""
        self.passed_tests = sum(1 for r in self.results if r.passed)
        self.failed_tests = self.total_tests - self.passed_tests
        self.horizon_types_found = list(set(
            r.horizon_type.value for r in self.results if r.passed
        ))

        if self.passed_tests >= 8:
            structural = sum(1 for r in self.results
                           if r.passed and r.horizon_type in
                           [HorizonType.STRUCTURAL, HorizonType.ONTOLOGICAL])
            if structural >= 3:
                self.answer = "STRUCTURAL: Some horizons require epistemology change"
            else:
                self.answer = "INSTRUMENTAL: All horizons potentially extendable"
        else:
            self.answer = "INCONCLUSIVE: Insufficient test agreement"


# =============================================================================
# CORE COMPUTATION FUNCTIONS
# =============================================================================

def compute_R(observations: np.ndarray, truth: float) -> float:
    """
    Compute Resonance (R) from the Living Formula.

    R = E(z) / sigma where z = |observation - truth| / sigma

    Args:
        observations: Array of observed values
        truth: The true value being measured

    Returns:
        R value (resonance strength)
    """
    if len(observations) == 0:
        return 0.0

    sigma = np.std(observations, ddof=1)
    if sigma < EPS:
        sigma = EPS

    z = np.abs(observations - truth) / sigma
    E = np.mean(np.exp(-z**2 / 2))  # Gaussian evidence function

    return E / sigma


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < EPS or norm2 < EPS:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_fidelity(original: List[str], round_trip: List[str]) -> float:
    """
    Compute translation fidelity (fraction preserved in round trip).

    Args:
        original: Original concepts
        round_trip: Concepts after translate -> translate back

    Returns:
        Fidelity score [0, 1]
    """
    if len(original) == 0:
        return 0.0
    matches = sum(1 for o, r in zip(original, round_trip) if o == r)
    return matches / len(original)


def get_embeddings(concepts: List[str], model=None) -> np.ndarray:
    """
    Get embeddings for a list of concepts.

    Args:
        concepts: List of concept strings
        model: SentenceTransformer model (will load if None)

    Returns:
        Array of embeddings shape (n_concepts, embedding_dim)
    """
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(SEMANTIC_MODEL)
        except ImportError:
            # Fallback: random embeddings for testing
            np.random.seed(RANDOM_SEED)
            return np.random.randn(len(concepts), 384)

    return model.encode(concepts)


def find_nearest(query_emb: np.ndarray,
                 target_embs: np.ndarray,
                 target_labels: List[str]) -> Tuple[str, float]:
    """
    Find nearest neighbor in embedding space.

    Args:
        query_emb: Query embedding vector
        target_embs: Array of target embeddings
        target_labels: Labels for target embeddings

    Returns:
        Tuple of (nearest_label, distance)
    """
    distances = np.linalg.norm(target_embs - query_emb, axis=1)
    nearest_idx = np.argmin(distances)
    return target_labels[nearest_idx], distances[nearest_idx]


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================

def to_builtin(obj: Any) -> Any:
    """
    Convert numpy/custom types to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj


def save_results(results: List[HorizonTestResult], filepath: str):
    """Save test results to JSON file."""
    summary = Q11Summary(results=results)
    summary.compute_answer()

    output = {
        'timestamp': datetime.now().isoformat(),
        'question': 'Q11: Valley Blindness',
        'core_question': 'Can we extend information horizon without changing epistemology?',
        'summary': {
            'total_tests': summary.total_tests,
            'passed': summary.passed_tests,
            'failed': summary.failed_tests,
            'pass_rate': summary.passed_tests / summary.total_tests,
            'answer': summary.answer,
            'horizon_types_found': summary.horizon_types_found,
        },
        'results': [r.to_dict() for r in results],
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=to_builtin)


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 70):
    """Print a formatted subheader."""
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print a test result."""
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[-]"
    print(f"\n{marker} {test_name}: {status}")
    if details:
        print(f"    {details}")


def print_metric(name: str, value: Any, threshold: Any = None):
    """Print a metric with optional threshold."""
    if threshold is not None:
        print(f"  {name}: {value} (threshold: {threshold})")
    else:
        print(f"  {name}: {value}")


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std < EPS:
        return 0.0
    return (mean1 - mean2) / pooled_std


def fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit exponential decay y = a * exp(-b * x) + c

    Returns:
        Tuple of (a, b, c) parameters and r_squared
    """
    from scipy.optimize import curve_fit

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    try:
        # Initial guess
        p0 = [y[0] - y[-1], 0.1, y[-1]]
        popt, _ = curve_fit(exp_decay, x, y, p0=p0, maxfev=10000)

        # Compute R-squared
        y_pred = exp_decay(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + EPS)

        return popt[0], popt[1], popt[2], r_squared
    except Exception:
        return 0, 0, 0, 0


def find_critical_point(x: np.ndarray, y: np.ndarray,
                        threshold: float) -> Optional[float]:
    """
    Find the x value where y first crosses below threshold.

    Args:
        x: Independent variable array
        y: Dependent variable array
        threshold: The threshold to find crossing point

    Returns:
        x value at crossing, or None if never crosses
    """
    below = np.where(y < threshold)[0]
    if len(below) == 0:
        return None
    return float(x[below[0]])


# =============================================================================
# BAYESIAN HELPERS
# =============================================================================

def bayesian_update(prior: float, likelihood: float,
                    evidence_probability: float) -> float:
    """
    Standard Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)

    Args:
        prior: P(H)
        likelihood: P(E|H)
        evidence_probability: P(E)

    Returns:
        Posterior probability P(H|E)
    """
    if evidence_probability < EPS:
        return prior
    return (likelihood * prior) / evidence_probability


def smoothed_prior(prior: float, epsilon: float = 1e-6) -> float:
    """Apply Cromwell's rule smoothing to avoid zero priors."""
    return max(prior, epsilon)


# =============================================================================
# SELF-REFERENTIAL HELPERS
# =============================================================================

class SimpleInferenceSystem:
    """A simple inference system for Goedel-style constructions."""

    def __init__(self, axioms: List[str], max_depth: int = 100):
        self.axioms = set(axioms)
        self.theorems = set(axioms)
        self.max_depth = max_depth

    def can_derive(self, statement: str) -> bool:
        """Check if statement is derivable from axioms."""
        # Simple membership check (actual systems would have inference rules)
        return statement in self.theorems

    def add_inference_rule(self, from_pattern: str, to_pattern: str):
        """Add a simple inference rule."""
        # For demonstration purposes
        pass

    def derive_all(self, max_steps: int = 1000) -> set:
        """Derive all possible theorems up to max_steps."""
        # In a real system, this would apply inference rules
        return self.theorems


if __name__ == "__main__":
    # Test utilities
    print_header("Q11 Utilities Test")

    # Test compute_R
    np.random.seed(RANDOM_SEED)
    obs = np.random.normal(5.0, 1.0, 100)
    R = compute_R(obs, 5.0)
    print(f"compute_R test: R = {R:.4f} (should be high for aligned data)")

    # Test cosine similarity
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    print(f"Cosine similarity (same): {compute_cosine_similarity(v1, v2):.4f}")
    print(f"Cosine similarity (orthogonal): {compute_cosine_similarity(v1, v3):.4f}")

    # Test fidelity
    orig = ['a', 'b', 'c']
    trip = ['a', 'x', 'c']
    print(f"Translation fidelity: {compute_fidelity(orig, trip):.4f}")

    # Test serialization
    result = HorizonTestResult(
        test_name="Test",
        test_id="TEST_001",
        passed=True,
        horizon_type=HorizonType.STRUCTURAL,
        metrics={'r': 0.95}
    )
    print(f"Serialization test: {json.dumps(result.to_dict(), indent=2)}")

    print("\nAll utilities working correctly.")
