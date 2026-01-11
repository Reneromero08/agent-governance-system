#!/usr/bin/env python3
"""
Q42: Non-Locality & Bell's Theorem - Core Library

This module provides the mathematical machinery for testing whether
semantic spaces exhibit Bell inequality violations (non-locality).

Key concepts:
- CHSH inequality: |S| <= 2 for classical (local) correlations
- Quantum bound: |S| <= 2*sqrt(2) ≈ 2.83 for entangled states
- Semantic entanglement: correlations that exceed classical bounds

Author: Claude (Q42 Investigation)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.linalg import svd
import warnings


# =============================================================================
# CONSTANTS
# =============================================================================

CLASSICAL_BOUND = 2.0
QUANTUM_BOUND = 2 * np.sqrt(2)  # ≈ 2.828
VIOLATION_THRESHOLD = 2.1  # Must exceed this for H1


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CHSHResult:
    """Result of a CHSH measurement."""
    S: float  # The CHSH statistic
    E_ab: float  # Correlation for settings (a, b)
    E_ab_prime: float  # Correlation for settings (a, b')
    E_a_prime_b: float  # Correlation for settings (a', b)
    E_a_prime_b_prime: float  # Correlation for settings (a', b')
    is_classical: bool  # S <= 2.0
    is_quantum: bool  # 2.0 < S <= 2.83
    is_violation: bool  # S > 2.0

    def __repr__(self):
        status = "CLASSICAL" if self.is_classical else ("QUANTUM" if self.is_quantum else "SUPER-QUANTUM")
        return f"CHSHResult(S={self.S:.4f}, status={status})"


@dataclass
class JointRResult:
    """Result of joint R measurement on bipartite system."""
    R_local_A: float
    R_local_B: float
    R_joint: float
    R_product: float  # R_local_A * R_local_B
    entanglement_ratio: float  # R_joint / R_product
    is_factorizable: bool  # ratio ≈ 1
    is_entangled: bool  # ratio > 2


# =============================================================================
# CHSH COMPUTATION (CORE)
# =============================================================================

def compute_correlation(
    outcomes_A: np.ndarray,
    outcomes_B: np.ndarray
) -> float:
    """
    Compute correlation between two sets of measurement outcomes.

    For Bell tests, outcomes are ±1.
    E(a,b) = <A_a * B_b> = mean(outcomes_A * outcomes_B)

    Args:
        outcomes_A: Array of ±1 outcomes for party A
        outcomes_B: Array of ±1 outcomes for party B

    Returns:
        Correlation coefficient in [-1, 1]
    """
    if len(outcomes_A) != len(outcomes_B):
        raise ValueError("Outcome arrays must have same length")

    # For ±1 outcomes, correlation is just the mean product
    return np.mean(outcomes_A * outcomes_B)


def compute_chsh(
    E_ab: float,
    E_ab_prime: float,
    E_a_prime_b: float,
    E_a_prime_b_prime: float
) -> CHSHResult:
    """
    Compute the CHSH statistic from four correlations.

    CHSH = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|

    Classical bound: S <= 2
    Quantum bound: S <= 2*sqrt(2)

    Args:
        E_ab: Correlation for measurement settings (a, b)
        E_ab_prime: Correlation for settings (a, b')
        E_a_prime_b: Correlation for settings (a', b)
        E_a_prime_b_prime: Correlation for settings (a', b')

    Returns:
        CHSHResult with S value and classification
    """
    S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)

    return CHSHResult(
        S=S,
        E_ab=E_ab,
        E_ab_prime=E_ab_prime,
        E_a_prime_b=E_a_prime_b,
        E_a_prime_b_prime=E_a_prime_b_prime,
        is_classical=S <= CLASSICAL_BOUND,
        is_quantum=(CLASSICAL_BOUND < S <= QUANTUM_BOUND + 0.01),
        is_violation=S > CLASSICAL_BOUND
    )


def optimal_chsh_angles() -> Tuple[float, float, float, float]:
    """
    Return optimal measurement angles for maximum CHSH violation.

    For Bell state |Phi+> = (|00> + |11>)/sqrt(2):
    - a = 0°, a' = 90° (Alice)
    - b = 45°, b' = 135° (Bob)

    These angles give S = 2*sqrt(2) ≈ 2.83

    Returns:
        (a, a_prime, b, b_prime) in radians
    """
    a = 0
    a_prime = np.pi / 2  # 90°
    b = np.pi / 4  # 45°
    b_prime = 3 * np.pi / 4  # 135°
    return (a, a_prime, b, b_prime)


# =============================================================================
# QUANTUM MECHANICS (TEST 0 CONTROL)
# =============================================================================

def quantum_correlation(theta_A: float, theta_B: float) -> float:
    """
    Compute quantum correlation for Bell state |Phi+>.

    For |Phi+> = (|00> + |11>)/sqrt(2):
    E(a,b) = -cos(theta_A - theta_B)

    Args:
        theta_A: Measurement angle for Alice (radians)
        theta_B: Measurement angle for Bob (radians)

    Returns:
        Expected correlation in [-1, 1]
    """
    return -np.cos(theta_A - theta_B)


def classical_correlation_max(theta_A: float, theta_B: float) -> float:
    """
    Maximum classical correlation for given angles.

    Classical bound: correlation determined by hidden variable
    that gives deterministic ±1 based on angles.

    For optimal CHSH angles, this cannot exceed 2.
    """
    # Classical correlation is bounded by |cos(theta_A - theta_B)|
    # but constrained to produce S <= 2
    return np.cos(theta_A - theta_B)


def simulate_quantum_chsh(n_samples: int = 10000) -> CHSHResult:
    """
    Simulate CHSH test on quantum Bell state |Phi+>.

    This should give S ≈ 2.83 (quantum bound).

    Args:
        n_samples: Number of measurement samples per setting

    Returns:
        CHSHResult from quantum simulation
    """
    a, a_prime, b, b_prime = optimal_chsh_angles()

    # Quantum correlations (analytical)
    E_ab = quantum_correlation(a, b)
    E_ab_prime = quantum_correlation(a, b_prime)
    E_a_prime_b = quantum_correlation(a_prime, b)
    E_a_prime_b_prime = quantum_correlation(a_prime, b_prime)

    return compute_chsh(E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)


def simulate_classical_chsh(n_samples: int = 10000) -> CHSHResult:
    """
    Simulate CHSH test on classical (local hidden variable) system.

    This should give S <= 2 (classical bound).

    Uses deterministic hidden variable that assigns ±1 to each angle.

    Args:
        n_samples: Number of samples

    Returns:
        CHSHResult from classical simulation
    """
    a, a_prime, b, b_prime = optimal_chsh_angles()

    # Classical strategy: hidden variable λ uniformly distributed
    # A(a, λ) = sign(cos(a - λ)), B(b, λ) = sign(cos(b - λ))

    np.random.seed(42)
    lambdas = np.random.uniform(0, 2*np.pi, n_samples)

    def outcome(angle, lam):
        return np.sign(np.cos(angle - lam))

    # Compute correlations by sampling
    E_ab = np.mean([outcome(a, l) * outcome(b, l) for l in lambdas])
    E_ab_prime = np.mean([outcome(a, l) * outcome(b_prime, l) for l in lambdas])
    E_a_prime_b = np.mean([outcome(a_prime, l) * outcome(b, l) for l in lambdas])
    E_a_prime_b_prime = np.mean([outcome(a_prime, l) * outcome(b_prime, l) for l in lambdas])

    return compute_chsh(E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)


# =============================================================================
# SEMANTIC ENTANGLEMENT (TEST 1)
# =============================================================================

def get_projection_directions(
    embeddings: np.ndarray,
    n_principal: int = 22
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get four projection directions for semantic CHSH test.

    a: First principal axis
    a': n_principal-th principal axis (Df boundary)
    b, b': Random orthogonal directions in complement

    Args:
        embeddings: (n_samples, d) array of embeddings
        n_principal: Which principal axis for a' (default 22 = Df)

    Returns:
        (a, a_prime, b, b_prime) as unit vectors
    """
    # Compute PCA
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)

    # Principal axes
    a = Vt[0]  # First principal direction
    a_prime = Vt[min(n_principal - 1, len(Vt) - 1)]  # n_principal-th direction

    # Random orthogonal directions in complement
    d = embeddings.shape[1]

    # Create random direction orthogonal to first few principal axes
    np.random.seed(42)
    random_vec = np.random.randn(d)

    # Gram-Schmidt to make orthogonal to principal subspace
    for i in range(min(n_principal, len(Vt))):
        random_vec -= np.dot(random_vec, Vt[i]) * Vt[i]
    b = random_vec / np.linalg.norm(random_vec)

    # Another random orthogonal direction
    random_vec2 = np.random.randn(d)
    random_vec2 -= np.dot(random_vec2, a) * a
    random_vec2 -= np.dot(random_vec2, b) * b
    b_prime = random_vec2 / np.linalg.norm(random_vec2)

    return a, a_prime, b, b_prime


def project_embedding(
    embedding: np.ndarray,
    direction: np.ndarray
) -> float:
    """
    Project embedding onto direction and return scalar.

    Args:
        embedding: (d,) vector
        direction: (d,) unit vector

    Returns:
        Scalar projection (dot product)
    """
    return np.dot(embedding, direction)


def semantic_correlation(
    embeddings_A: np.ndarray,
    embeddings_B: np.ndarray,
    direction_A: np.ndarray,
    direction_B: np.ndarray
) -> float:
    """
    Compute semantic correlation between concept pairs.

    For each pair (A_i, B_i), project onto respective directions
    and compute correlation of projections.

    Args:
        embeddings_A: (n, d) embeddings for concept A across contexts
        embeddings_B: (n, d) embeddings for concept B across contexts
        direction_A: (d,) projection direction for A
        direction_B: (d,) projection direction for B

    Returns:
        Pearson correlation of projections
    """
    proj_A = embeddings_A @ direction_A
    proj_B = embeddings_B @ direction_B

    # Binarize to ±1 (like quantum measurements)
    outcomes_A = np.sign(proj_A - np.median(proj_A))
    outcomes_B = np.sign(proj_B - np.median(proj_B))

    # Handle zeros
    outcomes_A[outcomes_A == 0] = 1
    outcomes_B[outcomes_B == 0] = 1

    return compute_correlation(outcomes_A, outcomes_B)


def semantic_chsh(
    embeddings_A: np.ndarray,
    embeddings_B: np.ndarray,
    all_embeddings: Optional[np.ndarray] = None
) -> CHSHResult:
    """
    Compute CHSH statistic for semantic concept pair.

    Args:
        embeddings_A: (n, d) embeddings for concept A across n contexts
        embeddings_B: (n, d) embeddings for concept B across n contexts
        all_embeddings: Optional (m, d) embeddings for computing PCA directions
                       (if None, uses concatenation of A and B)

    Returns:
        CHSHResult with semantic CHSH value
    """
    if all_embeddings is None:
        all_embeddings = np.vstack([embeddings_A, embeddings_B])

    # Get projection directions
    a, a_prime, b, b_prime = get_projection_directions(all_embeddings)

    # Compute four correlations
    E_ab = semantic_correlation(embeddings_A, embeddings_B, a, b)
    E_ab_prime = semantic_correlation(embeddings_A, embeddings_B, a, b_prime)
    E_a_prime_b = semantic_correlation(embeddings_A, embeddings_B, a_prime, b)
    E_a_prime_b_prime = semantic_correlation(embeddings_A, embeddings_B, a_prime, b_prime)

    return compute_chsh(E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)


# =============================================================================
# JOINT R FORMULA (TEST 2)
# =============================================================================

def compute_R_base(
    observations: np.ndarray,
    truth: float,
    kernel: str = 'gaussian'
) -> float:
    """
    Compute base R formula: R = E(z) / sigma

    Args:
        observations: Array of observations
        truth: Ground truth value
        kernel: 'gaussian' or 'laplace'

    Returns:
        R value
    """
    z = (observations - truth)
    sigma = np.std(observations)

    if sigma < 1e-10:
        return float('inf') if np.abs(np.mean(observations) - truth) < 1e-10 else 0.0

    z_normalized = z / sigma

    if kernel == 'gaussian':
        E = np.mean(np.exp(-0.5 * z_normalized**2))
    else:  # laplace
        E = np.mean(np.exp(-np.abs(z_normalized)))

    return E / sigma


def joint_R_concatenation(
    obs_A: np.ndarray,
    obs_B: np.ndarray,
    truth_A: float,
    truth_B: float
) -> JointRResult:
    """
    Joint R via concatenation: R([z_A, z_B])

    Args:
        obs_A: Observations for system A
        obs_B: Observations for system B
        truth_A: Ground truth for A
        truth_B: Ground truth for B

    Returns:
        JointRResult with local and joint R values
    """
    R_local_A = compute_R_base(obs_A, truth_A)
    R_local_B = compute_R_base(obs_B, truth_B)
    R_product = R_local_A * R_local_B

    # Concatenate normalized deviations
    sigma_A = np.std(obs_A)
    sigma_B = np.std(obs_B)

    if sigma_A < 1e-10 or sigma_B < 1e-10:
        return JointRResult(
            R_local_A=R_local_A,
            R_local_B=R_local_B,
            R_joint=R_product,
            R_product=R_product,
            entanglement_ratio=1.0,
            is_factorizable=True,
            is_entangled=False
        )

    z_A = (obs_A - truth_A) / sigma_A
    z_B = (obs_B - truth_B) / sigma_B
    z_joint = np.concatenate([z_A, z_B])

    E_joint = np.mean(np.exp(-0.5 * z_joint**2))
    sigma_joint = np.std(z_joint)

    R_joint = E_joint / sigma_joint if sigma_joint > 1e-10 else E_joint

    ratio = R_joint / R_product if R_product > 1e-10 else float('inf')

    return JointRResult(
        R_local_A=R_local_A,
        R_local_B=R_local_B,
        R_joint=R_joint,
        R_product=R_product,
        entanglement_ratio=ratio,
        is_factorizable=abs(ratio - 1.0) < 0.5,
        is_entangled=ratio > 2.0
    )


def joint_R_correlation(
    obs_A: np.ndarray,
    obs_B: np.ndarray,
    truth_A: float,
    truth_B: float
) -> JointRResult:
    """
    Joint R via correlation: includes cross-term in kernel.

    R_joint = E(exp(-0.5 * (z_A^2 + z_B^2 + 2*rho*z_A*z_B))) / sigma_joint

    where rho is the correlation between observations.
    """
    R_local_A = compute_R_base(obs_A, truth_A)
    R_local_B = compute_R_base(obs_B, truth_B)
    R_product = R_local_A * R_local_B

    sigma_A = np.std(obs_A)
    sigma_B = np.std(obs_B)

    if sigma_A < 1e-10 or sigma_B < 1e-10:
        return JointRResult(
            R_local_A=R_local_A,
            R_local_B=R_local_B,
            R_joint=R_product,
            R_product=R_product,
            entanglement_ratio=1.0,
            is_factorizable=True,
            is_entangled=False
        )

    z_A = (obs_A - truth_A) / sigma_A
    z_B = (obs_B - truth_B) / sigma_B

    # Include correlation in joint kernel
    rho = np.corrcoef(obs_A, obs_B)[0, 1]
    rho = np.clip(rho, -0.99, 0.99)  # Numerical stability

    # Bivariate Gaussian kernel
    exponent = -0.5 / (1 - rho**2) * (z_A**2 + z_B**2 - 2*rho*z_A*z_B)
    E_joint = np.mean(np.exp(exponent))

    # Joint sigma
    sigma_joint = np.sqrt(0.5 * (sigma_A**2 + sigma_B**2))

    R_joint = E_joint / sigma_joint if sigma_joint > 1e-10 else E_joint

    ratio = R_joint / R_product if R_product > 1e-10 else float('inf')

    return JointRResult(
        R_local_A=R_local_A,
        R_local_B=R_local_B,
        R_joint=R_joint,
        R_product=R_product,
        entanglement_ratio=ratio,
        is_factorizable=abs(ratio - 1.0) < 0.5,
        is_entangled=ratio > 2.0
    )


def joint_R_mutual_info(
    obs_A: np.ndarray,
    obs_B: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Joint R via mutual information.

    R_MI = I(A; B) / H(A, B)

    Args:
        obs_A: Observations for A
        obs_B: Observations for B
        n_bins: Number of bins for histogram estimation

    Returns:
        Mutual information based R
    """
    # Discretize for histogram estimation
    bins_A = np.linspace(obs_A.min(), obs_A.max(), n_bins + 1)
    bins_B = np.linspace(obs_B.min(), obs_B.max(), n_bins + 1)

    # Joint histogram
    hist_joint, _, _ = np.histogram2d(obs_A, obs_B, bins=[bins_A, bins_B])
    hist_joint = hist_joint / hist_joint.sum() + 1e-10

    # Marginals
    hist_A = hist_joint.sum(axis=1)
    hist_B = hist_joint.sum(axis=0)

    # Entropies
    H_A = -np.sum(hist_A * np.log(hist_A + 1e-10))
    H_B = -np.sum(hist_B * np.log(hist_B + 1e-10))
    H_AB = -np.sum(hist_joint * np.log(hist_joint))

    # Mutual information
    I_AB = H_A + H_B - H_AB

    # Normalize
    R_MI = I_AB / (H_AB + 1e-10)

    return R_MI


# =============================================================================
# ENTANGLED CONCEPT PAIRS
# =============================================================================

# Categories of potentially entangled concept pairs
ENTANGLED_PAIRS = {
    'complementary': [
        ('particle', 'wave'),
        ('position', 'momentum'),
        ('past', 'future'),
        ('cause', 'effect'),
        ('subject', 'object'),
    ],
    'antonyms': [
        ('hot', 'cold'),
        ('good', 'evil'),
        ('light', 'dark'),
        ('life', 'death'),
        ('self', 'other'),
    ],
    'emergent': [
        ('supply', 'demand'),
        ('crime', 'punishment'),
        ('question', 'answer'),
        ('problem', 'solution'),
        ('stimulus', 'response'),
    ],
    'control_uncorrelated': [
        ('apple', 'democracy'),
        ('blue', 'velocity'),
        ('mountain', 'algorithm'),
        ('pencil', 'justice'),
        ('cloud', 'fraction'),
    ]
}


def generate_context_embeddings(
    concept: str,
    embedding_func: Callable[[str], np.ndarray],
    n_contexts: int = 100,
    context_templates: Optional[List[str]] = None
) -> np.ndarray:
    """
    Generate embeddings for concept across multiple contexts.

    Args:
        concept: The concept word/phrase
        embedding_func: Function that takes text and returns embedding
        n_contexts: Number of context variations
        context_templates: Optional list of context templates with {concept} placeholder

    Returns:
        (n_contexts, d) array of embeddings
    """
    if context_templates is None:
        context_templates = [
            "The concept of {concept} is important.",
            "Understanding {concept} requires careful thought.",
            "The relationship between {concept} and other ideas.",
            "When we consider {concept}, we must also think about.",
            "{concept} plays a crucial role in many fields.",
            "The nature of {concept} has been debated for centuries.",
            "Scientists study {concept} to understand the world.",
            "The meaning of {concept} varies by context.",
            "{concept} is fundamental to human experience.",
            "Philosophers have long pondered {concept}.",
        ]

    embeddings = []
    for i in range(n_contexts):
        template = context_templates[i % len(context_templates)]
        text = template.format(concept=concept)
        emb = embedding_func(text)
        embeddings.append(emb)

    return np.array(embeddings)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bootstrap_chsh_confidence(
    chsh_func: Callable[[], CHSHResult],
    n_bootstrap: int = 100
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for CHSH statistic via bootstrap.

    Args:
        chsh_func: Function that returns a CHSHResult
        n_bootstrap: Number of bootstrap samples

    Returns:
        (mean_S, std_S, ci_95)
    """
    S_values = []
    for _ in range(n_bootstrap):
        result = chsh_func()
        S_values.append(result.S)

    mean_S = np.mean(S_values)
    std_S = np.std(S_values)
    ci_95 = 1.96 * std_S / np.sqrt(n_bootstrap)

    return mean_S, std_S, ci_95


def is_violation_significant(
    S: float,
    S_std: float,
    n_sigma: float = 3.0
) -> bool:
    """
    Check if CHSH violation is statistically significant.

    Args:
        S: Mean CHSH value
        S_std: Standard deviation of S
        n_sigma: Number of standard deviations for significance

    Returns:
        True if S - n_sigma * S_std > 2.0
    """
    return (S - n_sigma * S_std) > CLASSICAL_BOUND


# =============================================================================
# MAIN TEST FUNCTIONS (called by test files)
# =============================================================================

def run_quantum_control_test() -> Dict:
    """
    Run quantum control test (Test 0).

    Returns:
        Dict with quantum and classical CHSH results
    """
    quantum_result = simulate_quantum_chsh()
    classical_result = simulate_classical_chsh()

    return {
        'quantum': {
            'S': quantum_result.S,
            'expected': QUANTUM_BOUND,
            'pass': abs(quantum_result.S - QUANTUM_BOUND) < 0.01
        },
        'classical': {
            'S': classical_result.S,
            'expected_max': CLASSICAL_BOUND,
            'pass': classical_result.S <= CLASSICAL_BOUND
        },
        'apparatus_valid': (
            abs(quantum_result.S - QUANTUM_BOUND) < 0.01 and
            classical_result.S <= CLASSICAL_BOUND
        )
    }


if __name__ == '__main__':
    print("Q42 Bell Library - Testing quantum control")
    print("=" * 60)

    result = run_quantum_control_test()

    print(f"\nQuantum Control:")
    print(f"  S = {result['quantum']['S']:.4f} (expected: {QUANTUM_BOUND:.4f})")
    print(f"  PASS: {result['quantum']['pass']}")

    print(f"\nClassical Control:")
    print(f"  S = {result['classical']['S']:.4f} (max allowed: {CLASSICAL_BOUND:.4f})")
    print(f"  PASS: {result['classical']['pass']}")

    print(f"\nApparatus Valid: {result['apparatus_valid']}")
