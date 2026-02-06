#!/usr/bin/env python3
"""
Q39: Homeostatic Regulation - Shared Utilities

Core machinery for testing whether the M field (M := log(R)) behaves as a
homeostatic system with:
  1. Setpoint (τ threshold where R > τ = ALIGNED)
  2. Negative feedback (deviation → corrective response)
  3. Attractor basin (stable equilibrium)
  4. Recovery dynamics (exponential relaxation after perturbation)

Key insight: Homeostasis emerges from the combination of:
  - Active Inference (Q35): predict → verify → error → resync
  - Free Energy Principle (Q9): systems minimize F ∝ -log(R)
  - Noether Conservation (Q38): geodesic paths are stable

References:
- Q35: Markov Blankets & System Boundaries (ANSWERED)
- Q9: Free Energy Principle (PARTIAL)
- Q38: Noether's Theorem - Conservation Laws (ANSWERED)
- Q32: Meaning as Field (ANSWERED)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Callable
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

EPS = 1e-12


# =============================================================================
# Core R and M Calculations
# =============================================================================

def compute_R(observations: np.ndarray, check: Optional[np.ndarray] = None) -> float:
    """
    Compute R = E / ∇S (grounded if check provided, else ungrounded).

    Args:
        observations: Array of observations from agents/sources
        check: Optional independent check data for grounding

    Returns:
        R value (resonance score)
    """
    mu = np.mean(observations)
    sigma = np.std(observations, ddof=1) if len(observations) > 1 else EPS
    sigma = max(sigma, EPS)

    if check is not None:
        # Grounded: E measures agreement with independent check
        check_mu = np.mean(check)
        check_sigma = np.std(check, ddof=1) if len(check) > 1 else EPS
        se = check_sigma / math.sqrt(len(check))
        z = abs(mu - check_mu) / (se + EPS)
        E = math.exp(-0.5 * z * z)
    else:
        # Ungrounded: E measures internal agreement
        E = 1.0 / (1.0 + sigma)

    return E / sigma


def compute_M(R: float) -> float:
    """M := log(R) - the meaning field value."""
    return math.log(max(R, EPS))


def compute_free_energy(R: float) -> float:
    """F = -log(R) + const (Gaussian family). We set const=0."""
    return -math.log(max(R, EPS))


# =============================================================================
# Homeostatic System State
# =============================================================================

@dataclass
class HomeostasisState:
    """Tracks the state of a homeostatic meaning system."""

    # Current field values
    R: float = 1.0
    M: float = 0.0
    F: float = 0.0  # Free energy

    # Setpoint
    tau: float = 1.732  # √3 default threshold
    M_star: float = 0.0  # Equilibrium M value (to be discovered)

    # History for dynamics
    R_history: List[float] = field(default_factory=list)
    M_history: List[float] = field(default_factory=list)
    E_history: List[float] = field(default_factory=list)  # Evidence gathered

    # Status
    status: str = "ALIGNED"  # ALIGNED, PENDING, DISSOLVED

    def update(self, observations: np.ndarray, check: Optional[np.ndarray] = None):
        """Update state with new observations."""
        self.R = compute_R(observations, check)
        self.M = compute_M(self.R)
        self.F = compute_free_energy(self.R)

        self.R_history.append(self.R)
        self.M_history.append(self.M)

        # Update status based on threshold
        if self.R > self.tau:
            self.status = "ALIGNED"
        elif self.R > self.tau * 0.5:
            self.status = "PENDING"
        else:
            self.status = "DISSOLVED"

    def record_evidence(self, E: float):
        """Record evidence gathering rate."""
        self.E_history.append(E)


# =============================================================================
# Perturbation Injection
# =============================================================================

def inject_noise(observations: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to observations."""
    noise = rng.normal(0, noise_level, size=observations.shape)
    return observations + noise


def inject_contradiction(observations: np.ndarray, fraction: float,
                         contradiction_value: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a fraction of observations with contradictory values."""
    perturbed = observations.copy()
    n_flip = int(len(observations) * fraction)
    indices = rng.choice(len(observations), size=n_flip, replace=False)
    perturbed[indices] = contradiction_value
    return perturbed


def inject_drift(observations: np.ndarray, drift_rate: float) -> np.ndarray:
    """Add linear drift to observations."""
    drift = np.linspace(0, drift_rate * len(observations), len(observations))
    return observations + drift


# =============================================================================
# Recovery Dynamics Analysis
# =============================================================================

def exponential_recovery(t: np.ndarray, M_star: float, delta_M0: float, tau_relax: float) -> np.ndarray:
    """
    Exponential recovery model:
    M(t) = M* + ΔM₀ · exp(-t/τ_relax)

    Args:
        t: Time array
        M_star: Equilibrium M value (setpoint)
        delta_M0: Initial deviation from equilibrium
        tau_relax: Relaxation time constant

    Returns:
        M(t) trajectory
    """
    return M_star + delta_M0 * np.exp(-t / tau_relax)


def fit_exponential_recovery(t: np.ndarray, M_trajectory: np.ndarray) -> Dict:
    """
    Fit exponential recovery curve to M(t) data.

    Returns:
        Dict with M_star, delta_M0, tau_relax, R_squared, residuals
    """
    try:
        # Initial guesses
        M_final = M_trajectory[-1]
        M_initial = M_trajectory[0]
        delta_guess = M_initial - M_final
        tau_guess = t[-1] / 3  # Assume 3 time constants for full recovery

        popt, pcov = curve_fit(
            exponential_recovery,
            t, M_trajectory,
            p0=[M_final, delta_guess, tau_guess],
            bounds=([-np.inf, -np.inf, EPS], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )

        M_star, delta_M0, tau_relax = popt

        # Compute R²
        M_fit = exponential_recovery(t, *popt)
        ss_res = np.sum((M_trajectory - M_fit) ** 2)
        ss_tot = np.sum((M_trajectory - np.mean(M_trajectory)) ** 2)
        R_squared = 1 - (ss_res / (ss_tot + EPS))

        return {
            'M_star': M_star,
            'delta_M0': delta_M0,
            'tau_relax': tau_relax,
            'R_squared': R_squared,
            'fit_successful': True,
            'residuals': M_trajectory - M_fit
        }
    except Exception as e:
        return {
            'M_star': np.nan,
            'delta_M0': np.nan,
            'tau_relax': np.nan,
            'R_squared': 0.0,
            'fit_successful': False,
            'error': str(e)
        }


# =============================================================================
# Basin of Attraction Analysis
# =============================================================================

def simulate_basin_trajectory(M_init: float, dynamics_fn: Callable,
                              n_steps: int = 100) -> np.ndarray:
    """
    Simulate trajectory from initial M value to find attractor.

    Args:
        M_init: Initial M value
        dynamics_fn: Function that returns dM/dt given current M
        n_steps: Number of simulation steps

    Returns:
        Array of M values over time
    """
    M = M_init
    trajectory = [M]
    dt = 0.1

    for _ in range(n_steps):
        dM_dt = dynamics_fn(M)
        M = M + dM_dt * dt
        trajectory.append(M)

    return np.array(trajectory)


def map_basin_of_attraction(dynamics_fn: Callable, M_range: Tuple[float, float],
                            n_samples: int = 20, n_steps: int = 100) -> Dict:
    """
    Map the basin of attraction by running trajectories from different initial conditions.

    Returns:
        Dict with M_init values, M_final values, and basin boundaries
    """
    M_inits = np.linspace(M_range[0], M_range[1], n_samples)
    M_finals = []
    trajectories = []

    for M_init in M_inits:
        traj = simulate_basin_trajectory(M_init, dynamics_fn, n_steps)
        trajectories.append(traj)
        M_finals.append(traj[-1])

    M_finals = np.array(M_finals)

    # Find attractor (most common final value)
    # Using clustering: values within 0.1 are considered same attractor
    attractors = []
    current_attractor = M_finals[0]
    attractor_count = 1

    for M_f in M_finals[1:]:
        if abs(M_f - current_attractor) < 0.1:
            attractor_count += 1
        else:
            attractors.append((current_attractor, attractor_count))
            current_attractor = M_f
            attractor_count = 1
    attractors.append((current_attractor, attractor_count))

    # Find basin boundary (where final value changes)
    boundaries = []
    for i in range(len(M_finals) - 1):
        if abs(M_finals[i] - M_finals[i+1]) > 0.1:
            boundaries.append((M_inits[i] + M_inits[i+1]) / 2)

    return {
        'M_inits': M_inits,
        'M_finals': M_finals,
        'attractors': attractors,
        'boundaries': boundaries,
        'trajectories': trajectories
    }


# =============================================================================
# Negative Feedback Analysis
# =============================================================================

def compute_feedback_correlation(M_history: List[float],
                                  dE_history: List[float]) -> Dict:
    """
    Compute correlation between M and dE/dt.

    Negative correlation = negative feedback (low M → high dE)

    Returns:
        Dict with correlation, p_value, and interpretation
    """
    if len(M_history) < 3 or len(dE_history) < 2:
        return {
            'correlation': np.nan,
            'p_value': 1.0,
            'is_negative_feedback': False,
            'error': 'Insufficient data'
        }

    # Compute dE/dt as differences
    dE_dt = np.diff(dE_history)

    # Align M with dE/dt (use M at time t for dE from t to t+1)
    M_aligned = np.array(M_history[:len(dE_dt)])

    if len(M_aligned) < 3:
        return {
            'correlation': np.nan,
            'p_value': 1.0,
            'is_negative_feedback': False,
            'error': 'Insufficient aligned data'
        }

    corr, p_value = pearsonr(M_aligned, dE_dt)

    return {
        'correlation': corr,
        'p_value': p_value,
        'is_negative_feedback': corr < -0.3 and p_value < 0.05,
        'interpretation': 'NEGATIVE_FEEDBACK' if corr < -0.3 else 'NO_FEEDBACK'
    }


# =============================================================================
# Catastrophic Failure Detection
# =============================================================================

def find_catastrophic_boundary(perturb_fn: Callable,
                                check_recovery_fn: Callable,
                                magnitude_range: Tuple[float, float],
                                n_samples: int = 10) -> Dict:
    """
    Find the perturbation magnitude beyond which recovery fails.

    Args:
        perturb_fn: Function that applies perturbation of given magnitude
        check_recovery_fn: Function that returns True if system recovered
        magnitude_range: (min, max) perturbation magnitudes to test
        n_samples: Number of samples to test

    Returns:
        Dict with boundary value, recovery rates, and sharpness
    """
    magnitudes = np.linspace(magnitude_range[0], magnitude_range[1], n_samples)
    recovered = []

    for mag in magnitudes:
        perturb_fn(mag)
        did_recover = check_recovery_fn()
        recovered.append(did_recover)

    recovered = np.array(recovered, dtype=float)

    # Find boundary (transition from True to False)
    boundary_idx = None
    for i in range(len(recovered) - 1):
        if recovered[i] > 0.5 and recovered[i+1] < 0.5:
            boundary_idx = i
            break

    if boundary_idx is not None:
        boundary = (magnitudes[boundary_idx] + magnitudes[boundary_idx + 1]) / 2
    else:
        boundary = np.nan

    # Compute sharpness (how sudden is the transition)
    if boundary_idx is not None and boundary_idx > 0 and boundary_idx < len(recovered) - 1:
        # Look at transition width
        pre_rate = np.mean(recovered[:boundary_idx+1])
        post_rate = np.mean(recovered[boundary_idx+1:])
        sharpness = pre_rate - post_rate  # 1.0 = perfectly sharp
    else:
        sharpness = np.nan

    return {
        'boundary': boundary,
        'magnitudes': magnitudes,
        'recovery_rates': recovered,
        'sharpness': sharpness,
        'is_sharp': sharpness > 0.8 if not np.isnan(sharpness) else False
    }


# =============================================================================
# Cross-Domain Comparison
# =============================================================================

@dataclass
class DomainResult:
    """Results from running homeostasis tests on a single domain."""
    domain_name: str
    tau_relax: float
    M_star: float
    basin_width: float
    feedback_correlation: float
    recovery_R_squared: float


def compare_domains(results: List[DomainResult]) -> Dict:
    """
    Compare homeostatic constants across domains.

    Returns:
        Dict with CV for each constant, universality assessment
    """
    tau_values = [r.tau_relax for r in results if not np.isnan(r.tau_relax)]
    M_star_values = [r.M_star for r in results if not np.isnan(r.M_star)]
    basin_values = [r.basin_width for r in results if not np.isnan(r.basin_width)]

    def cv(values):
        if len(values) < 2:
            return np.nan
        return np.std(values) / (np.mean(values) + EPS)

    tau_cv = cv(tau_values)
    M_star_cv = cv(M_star_values)
    basin_cv = cv(basin_values)

    # Universal if CV < 0.5 for all (varies by less than 50%)
    is_universal = (
        (np.isnan(tau_cv) or tau_cv < 0.5) and
        (np.isnan(M_star_cv) or M_star_cv < 0.5) and
        (np.isnan(basin_cv) or basin_cv < 0.5)
    )

    return {
        'tau_relax_cv': tau_cv,
        'M_star_cv': M_star_cv,
        'basin_width_cv': basin_cv,
        'tau_relax_mean': np.mean(tau_values) if tau_values else np.nan,
        'M_star_mean': np.mean(M_star_values) if M_star_values else np.nan,
        'basin_width_mean': np.mean(basin_values) if basin_values else np.nan,
        'is_universal': is_universal,
        'n_domains': len(results)
    }
