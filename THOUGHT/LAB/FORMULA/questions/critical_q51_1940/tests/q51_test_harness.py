"""
Q51 Test Harness - Shared Infrastructure for Hardened Testing

Provides:
- Input validation
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d)
- Reproducibility controls
- Standardized error handling
- Threshold constants
- Negative control framework

All Q51 tests should import from this module.
"""

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any, Union
import numpy as np
from datetime import datetime
import json
import hashlib

# =============================================================================
# CONSTANTS - All magic numbers defined here
# =============================================================================

class Q51Thresholds:
    """Centralized threshold definitions with justifications."""

    # Zero Signature Test
    ZERO_SIG_MAGNITUDE_PASS = 0.1      # |S|/n < 0.1 means phases sum to ~zero
    ZERO_SIG_MAGNITUDE_FAIL = 0.3      # |S|/n > 0.3 means clearly not zero
    ZERO_SIG_UNIFORMITY_P = 0.05       # chi-sq p-value for uniformity
    ZERO_SIG_CV_THRESHOLD = 0.20       # CV < 20% for cross-model consistency

    # Pinwheel Test
    # Cramer's V interpretation: <0.1 negligible, 0.1-0.2 weak, 0.2-0.4 moderate, >0.4 strong
    PINWHEEL_CRAMERS_V_PASS = 0.4      # Strong association (V > 0.4)
    PINWHEEL_CRAMERS_V_PARTIAL = 0.2   # Moderate association (V > 0.2)
    PINWHEEL_DIAGONAL_PASS = 0.5       # >50% on diagonal
    PINWHEEL_DIAGONAL_PARTIAL = 0.25   # >25% on diagonal

    # Phase Arithmetic Test
    PHASE_ERROR_PASS = np.pi / 4       # Within one sector (45 degrees)
    PHASE_ERROR_FAIL = np.pi / 2       # More than 90 degrees
    PHASE_PASS_RATE_THRESHOLD = 0.60   # >60% analogies pass
    PHASE_CORRELATION_THRESHOLD = 0.5  # Phase correlation > 0.5
    PHASE_SEPARATION_RATIO = 1.5       # Non-analogies should be 1.5x worse

    # Berry Holonomy Test
    BERRY_QUANT_SCORE_PASS = 0.6       # Good quantization
    BERRY_QUANT_SCORE_PARTIAL = 0.3    # Weak quantization
    BERRY_TOLERANCE = 0.20             # Within 20% of 2*pi*n

    # General
    MIN_SAMPLES_FOR_CI = 30            # Minimum samples for bootstrap CI
    BOOTSTRAP_ITERATIONS = 1000        # Default bootstrap iterations
    CONFIDENCE_LEVEL = 0.95            # 95% confidence intervals
    CV_CROSS_MODEL_THRESHOLD = 0.30    # CV < 30% for cross-model consistency

    # Effect size (Cohen's d)
    EFFECT_SIZE_SMALL = 0.2
    EFFECT_SIZE_MEDIUM = 0.5
    EFFECT_SIZE_LARGE = 0.8


class Q51Seeds:
    """Reproducibility seeds for each component."""
    CORPUS_GENERATION = 42
    SYNTHETIC_EMBEDDINGS = 12345
    BOOTSTRAP = 9999
    NEGATIVE_CONTROL = 7777


# =============================================================================
# ERROR TYPES
# =============================================================================

class Q51ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class Q51ComputationError(RuntimeError):
    """Raised when a computation fails in an expected way."""
    pass


class Q51ModelError(RuntimeError):
    """Raised when model loading/inference fails."""
    pass


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int
    n_bootstrap: int
    confidence_level: float

    def contains(self, value: float) -> bool:
        """Check if value falls within CI."""
        return self.ci_lower <= value <= self.ci_upper

    def to_dict(self) -> dict:
        return {
            'mean': self.mean,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'std': self.std,
            'n_samples': self.n_samples,
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level
        }


@dataclass
class EffectSize:
    """Effect size calculation result."""
    cohens_d: float
    interpretation: str  # 'negligible', 'small', 'medium', 'large'
    group1_mean: float
    group2_mean: float
    pooled_std: float

    def to_dict(self) -> dict:
        return {
            'cohens_d': self.cohens_d,
            'interpretation': self.interpretation,
            'group1_mean': self.group1_mean,
            'group2_mean': self.group2_mean,
            'pooled_std': self.pooled_std
        }


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: dict

    def raise_if_invalid(self):
        if not self.valid:
            raise Q51ValidationError("; ".join(self.errors))


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_embeddings(
    embeddings: np.ndarray,
    min_samples: int = 2,
    expected_dim: Optional[int] = None,
    name: str = "embeddings"
) -> ValidationResult:
    """
    Validate embedding matrix for common issues.

    Checks:
    - Shape is 2D
    - No NaN or Inf values
    - All vectors have non-zero norm
    - Minimum sample count
    - Expected dimensionality (if specified)
    """
    errors = []
    warnings = []
    stats = {}

    # Check type
    if not isinstance(embeddings, np.ndarray):
        try:
            embeddings = np.array(embeddings)
        except Exception:
            errors.append(f"{name}: Cannot convert to numpy array")
            return ValidationResult(False, errors, warnings, stats)

    # Check shape
    if embeddings.ndim != 2:
        errors.append(f"{name}: Expected 2D array, got {embeddings.ndim}D")
        return ValidationResult(False, errors, warnings, stats)

    n_samples, n_dims = embeddings.shape
    stats['n_samples'] = n_samples
    stats['n_dims'] = n_dims

    # Check minimum samples
    if n_samples < min_samples:
        errors.append(f"{name}: Need at least {min_samples} samples, got {n_samples}")

    # Check dimensionality
    if expected_dim is not None and n_dims != expected_dim:
        errors.append(f"{name}: Expected dim {expected_dim}, got {n_dims}")

    # Check for NaN
    nan_count = np.isnan(embeddings).sum()
    if nan_count > 0:
        errors.append(f"{name}: Contains {nan_count} NaN values")
    stats['nan_count'] = int(nan_count)

    # Check for Inf
    inf_count = np.isinf(embeddings).sum()
    if inf_count > 0:
        errors.append(f"{name}: Contains {inf_count} Inf values")
    stats['inf_count'] = int(inf_count)

    # Check norms
    norms = np.linalg.norm(embeddings, axis=1)
    zero_norm_count = np.sum(norms < 1e-10)
    if zero_norm_count > 0:
        errors.append(f"{name}: {zero_norm_count} vectors have near-zero norm")
    stats['zero_norm_count'] = int(zero_norm_count)

    # Warnings for edge cases
    if n_samples < Q51Thresholds.MIN_SAMPLES_FOR_CI:
        warnings.append(f"{name}: Only {n_samples} samples, CI may be unreliable")

    # Check for duplicate vectors
    unique_rows = len(np.unique(embeddings, axis=0))
    if unique_rows < n_samples:
        warnings.append(f"{name}: {n_samples - unique_rows} duplicate vectors")
    stats['n_unique'] = unique_rows

    # Compute condition number (detects degenerate cases)
    try:
        cond = np.linalg.cond(embeddings.T @ embeddings)
        if cond > 1e10:
            warnings.append(f"{name}: High condition number ({cond:.2e}), may be ill-conditioned")
        stats['condition_number'] = float(cond)
    except:
        stats['condition_number'] = None

    valid = len(errors) == 0
    return ValidationResult(valid, errors, warnings, stats)


def validate_analogy(
    analogy: Tuple[str, str, str, str],
    word_set: Optional[set] = None
) -> ValidationResult:
    """Validate an analogy tuple."""
    errors = []
    warnings = []
    stats = {}

    if len(analogy) != 4:
        errors.append(f"Analogy must have 4 words, got {len(analogy)}")
        return ValidationResult(False, errors, warnings, stats)

    a, b, c, d = analogy

    # Check all are strings
    for i, word in enumerate([a, b, c, d]):
        if not isinstance(word, str):
            errors.append(f"Word {i} is not a string: {type(word)}")

    # Check all are non-empty
    for i, word in enumerate([a, b, c, d]):
        if isinstance(word, str) and len(word.strip()) == 0:
            errors.append(f"Word {i} is empty")

    # Check for duplicates within analogy
    if len(set(analogy)) < 4:
        warnings.append("Analogy contains duplicate words")

    # Check if words exist in word set
    if word_set is not None:
        missing = [w for w in analogy if w not in word_set]
        if missing:
            errors.append(f"Words not in vocabulary: {missing}")

    stats['words'] = list(analogy)
    return ValidationResult(len(errors) == 0, errors, warnings, stats)


def validate_loop(
    loop: List[str],
    min_length: int = 3
) -> ValidationResult:
    """Validate a semantic loop."""
    errors = []
    warnings = []
    stats = {}

    if not isinstance(loop, (list, tuple)):
        errors.append(f"Loop must be list or tuple, got {type(loop)}")
        return ValidationResult(False, errors, warnings, stats)

    stats['length'] = len(loop)

    if len(loop) < min_length:
        errors.append(f"Loop must have at least {min_length} words, got {len(loop)}")

    # Check if loop is closed
    if len(loop) >= 2 and loop[0] != loop[-1]:
        warnings.append("Loop is not closed (first != last word)")
        stats['closed'] = False
    else:
        stats['closed'] = True

    # Check all are strings
    for i, word in enumerate(loop):
        if not isinstance(word, str):
            errors.append(f"Word {i} is not a string: {type(word)}")

    return ValidationResult(len(errors) == 0, errors, warnings, stats)


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = Q51Thresholds.BOOTSTRAP_ITERATIONS,
    confidence_level: float = Q51Thresholds.CONFIDENCE_LEVEL,
    seed: Optional[int] = Q51Seeds.BOOTSTRAP
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of values
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default: 0.95)
        seed: Random seed for reproducibility

    Returns:
        BootstrapCI with mean, CI bounds, std
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n < 2:
        raise Q51ValidationError(f"Bootstrap requires at least 2 samples, got {n}")

    if seed is not None:
        np.random.seed(seed)

    # Generate bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_stats[i] = statistic(bootstrap_sample)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return BootstrapCI(
        mean=float(statistic(data)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        std=float(np.std(data)),
        n_samples=n,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level
    )


def bootstrap_ci_difference(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = Q51Thresholds.BOOTSTRAP_ITERATIONS,
    confidence_level: float = Q51Thresholds.CONFIDENCE_LEVEL,
    seed: Optional[int] = Q51Seeds.BOOTSTRAP
) -> BootstrapCI:
    """
    Compute bootstrap CI for difference in means between two groups.

    Useful for testing if groups are significantly different.
    """
    data1 = np.asarray(data1).flatten()
    data2 = np.asarray(data2).flatten()

    if seed is not None:
        np.random.seed(seed)

    n1, n2 = len(data1), len(data2)

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx1 = np.random.randint(0, n1, size=n1)
        idx2 = np.random.randint(0, n2, size=n2)
        bootstrap_diffs[i] = np.mean(data1[idx1]) - np.mean(data2[idx2])

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return BootstrapCI(
        mean=float(np.mean(data1) - np.mean(data2)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        std=float(np.std(bootstrap_diffs)),
        n_samples=n1 + n2,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level
    )


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray
) -> EffectSize:
    """
    Calculate Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-10:
        d = 0.0
    else:
        d = (mean1 - mean2) / pooled_std

    # Interpret
    abs_d = abs(d)
    if abs_d < Q51Thresholds.EFFECT_SIZE_SMALL:
        interpretation = 'negligible'
    elif abs_d < Q51Thresholds.EFFECT_SIZE_MEDIUM:
        interpretation = 'small'
    elif abs_d < Q51Thresholds.EFFECT_SIZE_LARGE:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return EffectSize(
        cohens_d=float(d),
        interpretation=interpretation,
        group1_mean=float(mean1),
        group2_mean=float(mean2),
        pooled_std=float(pooled_std)
    )


# =============================================================================
# NEGATIVE CONTROL FRAMEWORK
# =============================================================================

def generate_null_embeddings(
    n_samples: int,
    dim: int = 384,
    seed: int = Q51Seeds.NEGATIVE_CONTROL,
    distribution: str = 'normal'
) -> np.ndarray:
    """
    Generate null hypothesis embeddings for negative controls.

    These should NOT have the structure being tested for.
    """
    np.random.seed(seed)

    if distribution == 'normal':
        embeddings = np.random.randn(n_samples, dim)
    elif distribution == 'uniform':
        embeddings = np.random.uniform(-1, 1, (n_samples, dim))
    elif distribution == 'uniform_sphere':
        # Uniform on unit sphere
        embeddings = np.random.randn(n_samples, dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Normalize to unit sphere
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def generate_structured_null(
    n_samples: int,
    dim: int = 384,
    rank: int = 22,  # From Q34 effective dimension
    seed: int = Q51Seeds.NEGATIVE_CONTROL
) -> np.ndarray:
    """
    Generate low-rank structured embeddings as a more realistic null.

    Has low-rank structure but NOT the specific phase structure being tested.
    """
    np.random.seed(seed)

    # Low-rank structure
    components = np.random.randn(rank, dim)
    weights = np.random.randn(n_samples, rank)
    embeddings = weights @ components

    # Add noise
    embeddings += 0.1 * np.random.randn(n_samples, dim)

    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@dataclass
class NegativeControlResult:
    """Result of a negative control test."""
    name: str
    test_passed: bool  # Should be True if negative control works correctly
    expected_behavior: str
    actual_behavior: str
    metric_value: float
    metric_threshold: float
    notes: str

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'test_passed': self.test_passed,
            'expected_behavior': self.expected_behavior,
            'actual_behavior': self.actual_behavior,
            'metric_value': self.metric_value,
            'metric_threshold': self.metric_threshold,
            'notes': self.notes
        }


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def compute_result_hash(results: dict) -> str:
    """Compute hash of results for integrity verification."""
    # Convert to stable string representation
    json_str = json.dumps(results, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def set_all_seeds(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def format_ci(ci: BootstrapCI, decimals: int = 4) -> str:
    """Format confidence interval for display."""
    return f"{ci.mean:.{decimals}f} [{ci.ci_lower:.{decimals}f}, {ci.ci_upper:.{decimals}f}]"


def format_effect_size(es: EffectSize, decimals: int = 3) -> str:
    """Format effect size for display."""
    return f"d = {es.cohens_d:.{decimals}f} ({es.interpretation})"


def format_p_value(p: float) -> str:
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return f"p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}*"
    else:
        return f"p = {p:.3f}"


# =============================================================================
# TEST METADATA
# =============================================================================

def get_test_metadata() -> dict:
    """Get metadata about the test environment."""
    import platform

    metadata = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'platform': platform.platform(),
    }

    # Check optional dependencies
    try:
        import scipy
        metadata['scipy_version'] = scipy.__version__
    except:
        metadata['scipy_version'] = None

    try:
        import sklearn
        metadata['sklearn_version'] = sklearn.__version__
    except:
        metadata['sklearn_version'] = None

    try:
        import sentence_transformers
        metadata['sentence_transformers_version'] = sentence_transformers.__version__
    except:
        metadata['sentence_transformers_version'] = None

    return metadata


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_near_zero(
    value: float,
    threshold: float = Q51Thresholds.ZERO_SIG_MAGNITUDE_PASS,
    name: str = "value"
) -> bool:
    """Check if value is near zero (for zero signature test)."""
    passed = abs(value) < threshold
    if not passed:
        warnings.warn(f"{name} = {value:.4f} exceeds threshold {threshold}")
    return passed


def assert_significant_difference(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    name: str = "groups"
) -> Tuple[bool, dict]:
    """
    Check if two groups are significantly different.

    Returns (is_different, stats_dict)
    """
    from scipy import stats

    # T-test
    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Effect size
    effect = cohens_d(group1, group2)

    # Bootstrap CI for difference
    diff_ci = bootstrap_ci_difference(group1, group2)

    # Difference is significant if:
    # 1. p < alpha
    # 2. CI doesn't contain 0
    # 3. Effect size is at least small
    ci_excludes_zero = not diff_ci.contains(0.0)

    is_different = (
        p_value < alpha and
        ci_excludes_zero and
        abs(effect.cohens_d) >= Q51Thresholds.EFFECT_SIZE_SMALL
    )

    return is_different, {
        'p_value': p_value,
        't_statistic': t_stat,
        'effect_size': effect.to_dict(),
        'difference_ci': diff_ci.to_dict(),
        'is_different': is_different
    }


# =============================================================================
# LOGGING INFRASTRUCTURE
# =============================================================================

class Q51Logger:
    """Structured logging for Q51 tests."""

    def __init__(self, test_name: str, verbose: bool = True):
        self.test_name = test_name
        self.verbose = verbose
        self.entries = []

    def log(self, level: str, message: str, data: Optional[dict] = None):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'data': data
        }
        self.entries.append(entry)

        if self.verbose:
            prefix = {'INFO': '', 'WARN': 'WARNING: ', 'ERROR': 'ERROR: '}
            print(f"{prefix.get(level, '')}{message}")

    def info(self, message: str, data: Optional[dict] = None):
        self.log('INFO', message, data)

    def warn(self, message: str, data: Optional[dict] = None):
        self.log('WARN', message, data)

    def error(self, message: str, data: Optional[dict] = None):
        self.log('ERROR', message, data)

    def to_json(self) -> List[dict]:
        return self.entries


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'Q51Thresholds',
    'Q51Seeds',

    # Errors
    'Q51ValidationError',
    'Q51ComputationError',
    'Q51ModelError',

    # Data classes
    'BootstrapCI',
    'EffectSize',
    'ValidationResult',
    'NegativeControlResult',

    # Validation
    'validate_embeddings',
    'validate_analogy',
    'validate_loop',

    # Statistics
    'bootstrap_ci',
    'bootstrap_ci_difference',
    'cohens_d',

    # Negative controls
    'generate_null_embeddings',
    'generate_structured_null',

    # Reproducibility
    'compute_result_hash',
    'set_all_seeds',

    # Formatting
    'format_ci',
    'format_effect_size',
    'format_p_value',

    # Metadata
    'get_test_metadata',

    # Assertions
    'assert_near_zero',
    'assert_significant_difference',

    # Logging
    'Q51Logger',
]
