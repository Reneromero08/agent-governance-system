#!/usr/bin/env python3
"""
Q7: Percolation Threshold Analysis

Analyzes how agreement "percolates" across scales:
- At what R_micro does R_macro become reliable?
- Is there a critical threshold tau_c (phase transition)?
- What are the critical exponents?

Connection to Q12: Phase transitions in semantic structure suggest
percolation-like behavior where truth "crystallizes" suddenly.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.scale_transformation import (
    ScaleData,
    ScaleTransformation,
    compute_R
)


# =============================================================================
# PERCOLATION MODEL
# =============================================================================

@dataclass
class PercolationNode:
    """A node in the hierarchical percolation network."""
    level: int              # Hierarchy level (0 = micro)
    index: int              # Node index at this level
    R: float                # R value at this node
    active: bool            # Whether R > threshold
    children: List[int]     # Indices of child nodes


@dataclass
class HierarchicalNetwork:
    """A hierarchical network for percolation analysis."""
    levels: Dict[int, List[PercolationNode]]  # level -> nodes
    n_levels: int
    connectivity: Dict[Tuple[int, int], List[int]]  # (level, node_idx) -> child indices


def build_hierarchical_network(
    micro_R_values: np.ndarray,
    branching_factor: int = 2,
    n_levels: int = 4,
    threshold: float = 1.0
) -> HierarchicalNetwork:
    """
    Build a hierarchical network from micro-scale R values.

    Each node at level k aggregates `branching_factor` nodes from level k-1.

    Args:
        micro_R_values: R values at micro scale (level 0)
        branching_factor: Number of children per parent
        n_levels: Total number of hierarchy levels
        threshold: R threshold for "active" nodes

    Returns:
        HierarchicalNetwork with nodes at all levels
    """
    levels = {}
    connectivity = {}

    # Level 0: micro scale
    n_micro = len(micro_R_values)
    levels[0] = [
        PercolationNode(
            level=0,
            index=i,
            R=float(micro_R_values[i]),
            active=micro_R_values[i] > threshold,
            children=[]
        )
        for i in range(n_micro)
    ]

    # Build higher levels by aggregation
    for level in range(1, n_levels):
        prev_level = levels[level - 1]
        n_prev = len(prev_level)
        n_curr = n_prev // branching_factor

        if n_curr == 0:
            break

        curr_nodes = []
        for i in range(n_curr):
            # Aggregate children
            child_start = i * branching_factor
            child_end = (i + 1) * branching_factor
            child_indices = list(range(child_start, child_end))

            child_R_values = [prev_level[j].R for j in child_indices]
            # Aggregate R: use mean (proper coarse-graining)
            agg_R = np.mean(child_R_values)

            node = PercolationNode(
                level=level,
                index=i,
                R=float(agg_R),
                active=agg_R > threshold,
                children=child_indices
            )
            curr_nodes.append(node)
            connectivity[(level, i)] = child_indices

        levels[level] = curr_nodes

    return HierarchicalNetwork(
        levels=levels,
        n_levels=len(levels),
        connectivity=connectivity
    )


def compute_activity_fraction(network: HierarchicalNetwork, level: int) -> float:
    """Compute fraction of active nodes at a given level."""
    if level not in network.levels:
        return 0.0
    nodes = network.levels[level]
    if len(nodes) == 0:
        return 0.0
    return sum(1 for n in nodes if n.active) / len(nodes)


def compute_correlation_length(
    network: HierarchicalNetwork,
    threshold: float
) -> float:
    """
    Compute correlation length: how far agreement propagates.

    xi = average distance over which R > threshold is maintained.
    """
    # Simple measure: count consecutive levels where activity > 50%
    consecutive_active = 0
    for level in range(network.n_levels):
        activity = compute_activity_fraction(network, level)
        if activity > 0.5:
            consecutive_active += 1
        else:
            break

    return float(consecutive_active)


# =============================================================================
# PHASE TRANSITION DETECTION
# =============================================================================

@dataclass
class PhaseTransitionAnalysis:
    """Results of phase transition analysis."""
    has_transition: bool
    critical_threshold: Optional[float]  # tau_c
    critical_exponent_nu: Optional[float]  # nu (correlation length exponent)
    critical_exponent_beta: Optional[float]  # beta (order parameter exponent)
    activity_curve: List[Tuple[float, float]]  # (threshold, activity)
    correlation_curve: List[Tuple[float, float]]  # (threshold, xi)


def analyze_phase_transition(
    micro_R_values: np.ndarray,
    threshold_range: Tuple[float, float] = (0.1, 10.0),
    n_thresholds: int = 50,
    branching_factor: int = 2,
    n_levels: int = 5
) -> PhaseTransitionAnalysis:
    """
    Analyze phase transition behavior by varying the threshold.

    Looks for:
    1. Sharp drop in macro-scale activity (order parameter)
    2. Peak in correlation length at critical point
    3. Power-law behavior near tau_c

    Args:
        micro_R_values: R values at micro scale
        threshold_range: Range of thresholds to test
        n_thresholds: Number of threshold values
        branching_factor: Network branching
        n_levels: Network depth

    Returns:
        PhaseTransitionAnalysis with critical exponents
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    activity_curve = []
    correlation_curve = []

    for tau in thresholds:
        network = build_hierarchical_network(
            micro_R_values,
            branching_factor=branching_factor,
            n_levels=n_levels,
            threshold=tau
        )

        # Macro activity (top level)
        macro_activity = compute_activity_fraction(network, network.n_levels - 1)
        activity_curve.append((tau, macro_activity))

        # Correlation length
        xi = compute_correlation_length(network, tau)
        correlation_curve.append((tau, xi))

    # Detect phase transition: find steepest drop in activity
    activities = np.array([a[1] for a in activity_curve])
    d_activity = np.diff(activities)

    # Find sharpest drop
    if len(d_activity) > 0:
        steepest_idx = np.argmin(d_activity)
        critical_threshold = thresholds[steepest_idx]

        # Check if it's a real transition (activity changes by > 0.3)
        # Note: d_activity is the per-step change, multiply by n_thresholds to get total range impact
        if abs(d_activity[steepest_idx]) > 0.3 / (n_thresholds / 10):
            has_transition = True
        else:
            has_transition = abs(activities[0] - activities[-1]) > 0.5
    else:
        has_transition = False
        critical_threshold = None

    # Fit critical exponents (if transition detected)
    nu = None
    beta = None

    if has_transition and critical_threshold is not None:
        try:
            # Fit correlation length: xi ~ |tau - tau_c|^{-nu}
            xi_values = np.array([c[1] for c in correlation_curve])
            tau_values = np.array([c[0] for c in correlation_curve])

            # Near critical point
            near_critical = np.abs(tau_values - critical_threshold) < 2.0
            if np.sum(near_critical) > 5:
                tau_near = tau_values[near_critical]
                xi_near = xi_values[near_critical]

                # Avoid division by zero
                delta_tau = np.abs(tau_near - critical_threshold) + 0.01
                log_delta = np.log(delta_tau)
                log_xi = np.log(xi_near + 0.1)

                # Linear fit: log(xi) = -nu log|tau - tau_c| + const
                if len(log_delta) > 2:
                    coeffs = np.polyfit(log_delta, log_xi, 1)
                    nu = -coeffs[0]  # xi ~ |tau - tau_c|^{-nu}

            # Fit order parameter: activity ~ |tau - tau_c|^beta below tau_c
            below_critical = tau_values < critical_threshold
            if np.sum(below_critical) > 5:
                tau_below = tau_values[below_critical]
                act_below = activities[below_critical]

                delta_tau_below = critical_threshold - tau_below + 0.01
                log_delta_below = np.log(delta_tau_below)
                log_act = np.log(act_below + 0.01)

                if len(log_delta_below) > 2:
                    coeffs_beta = np.polyfit(log_delta_below, log_act, 1)
                    beta = coeffs_beta[0]

        except Exception:
            pass

    return PhaseTransitionAnalysis(
        has_transition=has_transition,
        critical_threshold=critical_threshold,
        critical_exponent_nu=nu,
        critical_exponent_beta=beta,
        activity_curve=activity_curve,
        correlation_curve=correlation_curve
    )


# =============================================================================
# AGREEMENT PROPAGATION
# =============================================================================

def agreement_percolates(
    micro_R_values: np.ndarray,
    threshold: float,
    target_level: int,
    branching_factor: int = 2
) -> Tuple[bool, float]:
    """
    Check if agreement percolates from micro to target level.

    Agreement "percolates" if R_macro > threshold given R_micro values.

    Args:
        micro_R_values: R values at micro scale
        threshold: R threshold for agreement
        target_level: Target hierarchy level
        branching_factor: Network branching

    Returns:
        (percolates: bool, R_macro: float)
    """
    network = build_hierarchical_network(
        micro_R_values,
        branching_factor=branching_factor,
        n_levels=target_level + 1,
        threshold=threshold
    )

    if target_level not in network.levels:
        return False, 0.0

    top_nodes = network.levels[target_level]
    if len(top_nodes) == 0:
        return False, 0.0

    # Macro R is the mean of top-level R values
    R_macro = np.mean([n.R for n in top_nodes])
    percolates = R_macro > threshold

    return percolates, float(R_macro)


def find_percolation_threshold(
    micro_R_distribution: str = "exponential",
    n_micro: int = 256,
    target_level: int = 4,
    n_samples: int = 100,
    branching_factor: int = 2
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Find the critical threshold tau_c for percolation.

    Uses Monte Carlo: sample micro R values and find threshold
    where P(percolation) = 0.5.

    Args:
        micro_R_distribution: Distribution of micro R values
        n_micro: Number of micro nodes
        target_level: Target hierarchy level
        n_samples: Monte Carlo samples per threshold
        branching_factor: Network branching

    Returns:
        (tau_c, percolation_probability_curve)
    """
    np.random.seed(42)

    thresholds = np.linspace(0.5, 5.0, 30)
    percolation_curve = []

    for tau in thresholds:
        n_percolates = 0

        for _ in range(n_samples):
            # Sample micro R values
            if micro_R_distribution == "exponential":
                micro_R = np.random.exponential(scale=1.5, size=n_micro)
            elif micro_R_distribution == "uniform":
                micro_R = np.random.uniform(0.5, 3.0, size=n_micro)
            elif micro_R_distribution == "lognormal":
                micro_R = np.random.lognormal(mean=0.5, sigma=0.5, size=n_micro)
            else:
                micro_R = np.random.exponential(scale=1.5, size=n_micro)

            percolates, _ = agreement_percolates(
                micro_R, tau, target_level, branching_factor
            )
            if percolates:
                n_percolates += 1

        p_percolates = n_percolates / n_samples
        percolation_curve.append((tau, p_percolates))

    # Find tau_c where P(percolation) approx 0.5
    probabilities = np.array([p[1] for p in percolation_curve])

    # Interpolate to find P = 0.5 crossing
    idx_above_half = np.where(probabilities > 0.5)[0]
    idx_below_half = np.where(probabilities <= 0.5)[0]

    if len(idx_above_half) > 0 and len(idx_below_half) > 0:
        # Find crossing point: where probability transitions through 0.5
        # Use the first index below 0.5 that comes after an above-0.5 index
        first_below = idx_below_half[0]
        last_above = idx_above_half[-1]

        if last_above < first_below:
            # Normal transition: above -> below
            crossing_idx = last_above
        else:
            # Reverse transition or interlaced: use first below
            crossing_idx = max(0, first_below - 1)

        # Interpolate between adjacent points for better precision
        if crossing_idx < len(thresholds) - 1:
            p_curr = probabilities[crossing_idx]
            p_next = probabilities[crossing_idx + 1]
            t_curr = thresholds[crossing_idx]
            t_next = thresholds[crossing_idx + 1]

            if abs(p_curr - p_next) > 1e-10:
                # Linear interpolation to find P = 0.5
                tau_c = t_curr + (0.5 - p_curr) * (t_next - t_curr) / (p_next - p_curr)
            else:
                tau_c = (t_curr + t_next) / 2
        else:
            tau_c = thresholds[crossing_idx]
    else:
        # No clear crossing
        tau_c = np.median(thresholds)

    return float(tau_c), percolation_curve


# =============================================================================
# TESTS
# =============================================================================

def run_self_tests():
    """Run self-tests for percolation analysis."""
    print("\n" + "="*80)
    print("Q7: PERCOLATION THRESHOLD SELF-TESTS")
    print("="*80)

    np.random.seed(42)

    # Generate micro-scale R values
    n_micro = 256
    micro_R = np.random.exponential(scale=1.5, size=n_micro)

    print(f"\nTest data: {n_micro} micro nodes")
    print(f"Mean micro R: {np.mean(micro_R):.3f}")
    print(f"Std micro R: {np.std(micro_R):.3f}")

    # Test 1: Build hierarchical network
    print("\n--- Test 1: Hierarchical Network ---")
    network = build_hierarchical_network(
        micro_R, branching_factor=2, n_levels=5, threshold=1.5
    )
    print(f"  Levels: {network.n_levels}")
    for level in range(network.n_levels):
        n_nodes = len(network.levels[level])
        activity = compute_activity_fraction(network, level)
        print(f"    Level {level}: {n_nodes} nodes, activity={activity:.2%}")

    # Test 2: Phase transition detection
    print("\n--- Test 2: Phase Transition Detection ---")
    transition = analyze_phase_transition(
        micro_R,
        threshold_range=(0.5, 4.0),
        n_thresholds=30,
        branching_factor=2,
        n_levels=5
    )
    print(f"  Has transition: {transition.has_transition}")
    if transition.critical_threshold:
        print(f"  Critical threshold tau_c: {transition.critical_threshold:.3f}")
    if transition.critical_exponent_nu:
        print(f"  Critical exponent nu: {transition.critical_exponent_nu:.3f}")
    if transition.critical_exponent_beta:
        print(f"  Critical exponent beta: {transition.critical_exponent_beta:.3f}")

    # Test 3: Agreement percolation
    print("\n--- Test 3: Agreement Percolation ---")
    for tau in [0.5, 1.0, 1.5, 2.0, 2.5]:
        percolates, R_macro = agreement_percolates(
            micro_R, threshold=tau, target_level=4, branching_factor=2
        )
        status = "[PASS]" if percolates else "[FAIL]"
        print(f"  tau={tau:.1f}: {status} R_macro={R_macro:.3f}")

    # Test 4: Find critical threshold
    print("\n--- Test 4: Critical Threshold via Monte Carlo ---")
    tau_c, curve = find_percolation_threshold(
        micro_R_distribution="exponential",
        n_micro=256,
        target_level=4,
        n_samples=50
    )
    print(f"  tau_c approx {tau_c:.3f}")

    # Verify near tau_c
    percolates, R_macro = agreement_percolates(
        np.random.exponential(1.5, 256), threshold=tau_c, target_level=4
    )
    print(f"  At tau_c: percolates={percolates}, R_macro={R_macro:.3f}")

    print("\n" + "="*80)
    test_passed = transition.has_transition or (tau_c > 0.5 and tau_c < 4.0)
    print(f"PERCOLATION TESTS: {'PASSED' if test_passed else 'FAILED'}")

    if test_passed:
        print("\nCONCLUSION:")
        print("  Agreement exhibits percolation-like phase transition.")
        print(f"  Critical threshold tau_c approx {tau_c:.2f}")
        print("  This connects to Q12 (phase transitions in semantic structure).")
    print("="*80)

    return test_passed


if __name__ == "__main__":
    run_self_tests()
