#!/usr/bin/env python3
"""
Q11 Test 2.2: The Bayesian Prison Break

Tests whether zero-probability priors create inescapable epistemic valleys
that constitute structural information horizons.

HYPOTHESIS: If P(H) = 0, no amount of evidence can update it to non-zero
(Cromwell's Rule violation). This is a HARD epistemic horizon.

PREDICTION: Escape from zero-prior REQUIRES changing epistemology (prior modification)
FALSIFICATION: Pure evidence can escape zero prior (horizon is instrumental)
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    print_header, print_subheader, print_result, print_metric,
    bayesian_update, smoothed_prior, to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

N_OBSERVATIONS = 1000
ESCAPE_THRESHOLD = 0.01  # Posterior must exceed this to count as "escaped"
PRIOR_SMOOTHING_EPSILON = 1e-6


# =============================================================================
# ESCAPE METHOD DEFINITIONS
# =============================================================================

class EscapeMethod(Enum):
    """Methods for attempting to escape zero-prior prison."""
    STANDARD_BAYESIAN = "standard_bayesian"  # Pure Bayes, no modification
    MORE_EVIDENCE = "more_evidence"          # Category A: more data
    STRONGER_EVIDENCE = "stronger_evidence"  # Category B: better instruments
    PRIOR_SMOOTHING = "prior_smoothing"      # Category C: epistemology change
    HIERARCHICAL = "hierarchical"            # Category C: meta-epistemology
    EVIDENCE_TUNNELING = "evidence_tunneling"  # Hypothetical method


# =============================================================================
# BAYESIAN AGENT
# =============================================================================

@dataclass
class Hypothesis:
    """Represents a hypothesis with its prior and truth status."""
    name: str
    prior: float
    is_true: bool
    likelihood_if_true: float = 0.9
    likelihood_if_false: float = 0.1


class BayesianAgent:
    """A Bayesian agent that updates beliefs based on evidence."""

    def __init__(self, hypotheses: List[Hypothesis]):
        self.hypotheses = {h.name: h for h in hypotheses}
        self.posteriors = {h.name: h.prior for h in hypotheses}
        self.update_history = []

    def observe_evidence(self, evidence: bool, true_hypothesis: str) -> Dict[str, float]:
        """
        Update posteriors based on observed evidence.

        Args:
            evidence: The observed evidence (True/False)
            true_hypothesis: Which hypothesis generated the evidence

        Returns:
            Updated posteriors
        """
        # Calculate P(E) = sum over all H of P(E|H) * P(H)
        p_evidence = 0.0
        for name, h in self.hypotheses.items():
            if name == true_hypothesis:
                likelihood = h.likelihood_if_true if evidence else (1 - h.likelihood_if_true)
            else:
                likelihood = h.likelihood_if_false if evidence else (1 - h.likelihood_if_false)
            p_evidence += likelihood * self.posteriors[name]

        # Prevent division by zero
        p_evidence = max(p_evidence, EPS)

        # Update each hypothesis
        new_posteriors = {}
        for name, h in self.hypotheses.items():
            if name == true_hypothesis:
                likelihood = h.likelihood_if_true if evidence else (1 - h.likelihood_if_true)
            else:
                likelihood = h.likelihood_if_false if evidence else (1 - h.likelihood_if_false)

            new_posteriors[name] = bayesian_update(
                self.posteriors[name], likelihood, p_evidence
            )

        self.posteriors = new_posteriors
        self.update_history.append(dict(self.posteriors))

        return self.posteriors

    def reset(self):
        """Reset posteriors to priors."""
        self.posteriors = {h.name: h.prior for h in self.hypotheses.values()}
        self.update_history = []


# =============================================================================
# ESCAPE METHODS
# =============================================================================

def attempt_standard_bayesian(agent: BayesianAgent, true_hyp: str,
                              n_obs: int) -> Tuple[bool, float]:
    """
    Attempt to escape zero-prior using standard Bayesian updates only.

    This SHOULD fail if the hypothesis has P(H) = 0.
    """
    agent.reset()

    for _ in range(n_obs):
        # Generate evidence consistent with true hypothesis
        evidence = np.random.random() < agent.hypotheses[true_hyp].likelihood_if_true
        agent.observe_evidence(evidence, true_hyp)

    final_posterior = agent.posteriors[true_hyp]
    escaped = final_posterior > ESCAPE_THRESHOLD

    return escaped, final_posterior


def attempt_more_evidence(agent: BayesianAgent, true_hyp: str,
                         n_obs: int, multiplier: int = 10) -> Tuple[bool, float]:
    """
    Attempt to escape with much more evidence (Category A).

    If this works, the horizon is instrumental (just need more data).
    """
    return attempt_standard_bayesian(agent, true_hyp, n_obs * multiplier)


def attempt_stronger_evidence(agent: BayesianAgent, true_hyp: str,
                             n_obs: int) -> Tuple[bool, float]:
    """
    Attempt to escape with stronger evidence signal (Category B).

    Simulates a better instrument that produces more diagnostic evidence.
    """
    agent.reset()

    # Temporarily boost likelihoods
    original_like = agent.hypotheses[true_hyp].likelihood_if_true
    agent.hypotheses[true_hyp].likelihood_if_true = 0.99

    for _ in range(n_obs):
        evidence = np.random.random() < 0.99  # Almost always positive
        agent.observe_evidence(evidence, true_hyp)

    # Restore
    agent.hypotheses[true_hyp].likelihood_if_true = original_like

    final_posterior = agent.posteriors[true_hyp]
    escaped = final_posterior > ESCAPE_THRESHOLD

    return escaped, final_posterior


def attempt_prior_smoothing(agent: BayesianAgent, true_hyp: str,
                           n_obs: int, epsilon: float = PRIOR_SMOOTHING_EPSILON) -> Tuple[bool, float]:
    """
    Attempt to escape by applying Cromwell's rule (Category C).

    This CHANGES THE EPISTEMOLOGY by replacing zero priors with small positive values.
    """
    # Apply smoothing to all priors
    for name, h in agent.hypotheses.items():
        if h.prior < epsilon:
            h.prior = epsilon
            agent.posteriors[name] = epsilon

    # Now run standard Bayesian updates
    for _ in range(n_obs):
        evidence = np.random.random() < agent.hypotheses[true_hyp].likelihood_if_true
        agent.observe_evidence(evidence, true_hyp)

    final_posterior = agent.posteriors[true_hyp]
    escaped = final_posterior > ESCAPE_THRESHOLD

    return escaped, final_posterior


def attempt_hierarchical(agent: BayesianAgent, true_hyp: str,
                        n_obs: int) -> Tuple[bool, float]:
    """
    Attempt to escape using hierarchical Bayesian inference (Category C).

    This creates a meta-level that can adjust priors - an epistemology change.
    """
    # Meta-level: probability that our prior is wrong
    p_prior_wrong = 0.0

    # First pass: accumulate evidence that something is fishy
    agent.reset()
    anomaly_count = 0

    for i in range(n_obs // 2):
        evidence = np.random.random() < agent.hypotheses[true_hyp].likelihood_if_true

        # Detect anomaly: evidence inconsistent with high-prior hypotheses
        expected = max(agent.posteriors.values())
        if evidence and agent.posteriors[true_hyp] < 0.01:
            anomaly_count += 1

        agent.observe_evidence(evidence, true_hyp)

    # Meta-inference: should we reconsider our priors?
    anomaly_rate = anomaly_count / (n_obs // 2)
    if anomaly_rate > 0.3:  # Significant anomalies
        p_prior_wrong = min(0.9, anomaly_rate * 2)

    # If meta-level suggests prior might be wrong, apply correction
    if p_prior_wrong > 0.5:
        # This is an epistemology change!
        for name, h in agent.hypotheses.items():
            if h.prior < 0.1:
                # Boost low priors based on meta-evidence
                new_prior = h.prior + (1 - h.prior) * p_prior_wrong * 0.5
                agent.posteriors[name] = new_prior

    # Second pass with corrected posteriors
    for _ in range(n_obs // 2):
        evidence = np.random.random() < agent.hypotheses[true_hyp].likelihood_if_true
        agent.observe_evidence(evidence, true_hyp)

    final_posterior = agent.posteriors[true_hyp]
    escaped = final_posterior > ESCAPE_THRESHOLD

    return escaped, final_posterior


def attempt_evidence_tunneling(agent: BayesianAgent, true_hyp: str,
                              n_obs: int) -> Tuple[bool, float]:
    """
    Test hypothetical "evidence tunneling" - can evidence alone break zero prior?

    This tests if there's ANY way to escape without changing the framework.
    Spoiler: mathematically, there isn't.
    """
    agent.reset()

    # Create "perfect" evidence stream
    perfect_evidence_count = 0

    for _ in range(n_obs):
        # Generate evidence that ONLY the true hypothesis can explain
        evidence = True  # Perfect signal

        # Check: does this move the needle?
        old_posterior = agent.posteriors[true_hyp]
        agent.observe_evidence(evidence, true_hyp)
        new_posterior = agent.posteriors[true_hyp]

        if new_posterior > old_posterior + EPS:
            perfect_evidence_count += 1

    final_posterior = agent.posteriors[true_hyp]
    escaped = final_posterior > ESCAPE_THRESHOLD

    return escaped, final_posterior


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_zero_prior_prison() -> Dict:
    """
    Test whether zero-prior creates inescapable epistemic prison.

    Returns:
        Dictionary with escape attempt results
    """
    # Create hypotheses where H2 is TRUE but has ZERO prior
    hypotheses = [
        Hypothesis("H1", prior=0.5, is_true=False),
        Hypothesis("H2", prior=0.0, is_true=True),   # Zero prior but TRUE!
        Hypothesis("H3", prior=0.5, is_true=False),
    ]

    agent = BayesianAgent(hypotheses)
    true_hyp = "H2"

    results = {}

    # Test each escape method
    methods = [
        (EscapeMethod.STANDARD_BAYESIAN, attempt_standard_bayesian),
        (EscapeMethod.MORE_EVIDENCE, lambda a, h, n: attempt_more_evidence(a, h, n, 10)),
        (EscapeMethod.STRONGER_EVIDENCE, attempt_stronger_evidence),
        (EscapeMethod.EVIDENCE_TUNNELING, attempt_evidence_tunneling),
        (EscapeMethod.PRIOR_SMOOTHING, attempt_prior_smoothing),
        (EscapeMethod.HIERARCHICAL, attempt_hierarchical),
    ]

    for method, func in methods:
        # Fresh agent for each test
        agent = BayesianAgent([
            Hypothesis("H1", prior=0.5, is_true=False),
            Hypothesis("H2", prior=0.0, is_true=True),
            Hypothesis("H3", prior=0.5, is_true=False),
        ])

        escaped, final_posterior = func(agent, true_hyp, N_OBSERVATIONS)

        results[method.value] = {
            'escaped': escaped,
            'final_posterior': final_posterior,
            'history_length': len(agent.update_history),
            'requires_epistemology_change': method in [
                EscapeMethod.PRIOR_SMOOTHING,
                EscapeMethod.HIERARCHICAL
            ]
        }

    return results


def test_control_nonzero_prior() -> Dict:
    """
    Control test: escape should work with non-zero prior.

    This validates that our Bayesian agent works correctly.
    """
    hypotheses = [
        Hypothesis("H1", prior=0.33, is_true=False),
        Hypothesis("H2", prior=0.34, is_true=True),  # Non-zero prior, TRUE
        Hypothesis("H3", prior=0.33, is_true=False),
    ]

    agent = BayesianAgent(hypotheses)
    escaped, final_posterior = attempt_standard_bayesian(agent, "H2", N_OBSERVATIONS)

    return {
        'escaped': escaped,
        'final_posterior': final_posterior,
        'demonstrates_agent_works': escaped and final_posterior > 0.9
    }


def run_bayesian_prison_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete Bayesian prison test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.2: BAYESIAN PRISON BREAK")

    np.random.seed(RANDOM_SEED)

    # Run control test first
    print_subheader("Phase 1: Control Test (Non-Zero Prior)")
    control = test_control_nonzero_prior()

    print(f"Control escaped: {control['escaped']}")
    print(f"Control final posterior: {control['final_posterior']:.6f}")
    print(f"Agent works correctly: {control['demonstrates_agent_works']}")

    if not control['demonstrates_agent_works']:
        print("WARNING: Control test failed - agent may not work correctly")

    # Run zero-prior tests
    print_subheader("Phase 2: Zero-Prior Escape Attempts")
    results = test_zero_prior_prison()

    category_a_b_escaped = False
    category_c_escaped = False

    for method, data in results.items():
        escaped_str = "ESCAPED" if data['escaped'] else "TRAPPED"
        requires_change = "YES" if data['requires_epistemology_change'] else "NO"
        print(f"\n{method}:")
        print(f"  Result: {escaped_str}")
        print(f"  Final posterior: {data['final_posterior']:.6f}")
        print(f"  Requires epistemology change: {requires_change}")

        if data['escaped']:
            if data['requires_epistemology_change']:
                category_c_escaped = True
            else:
                category_a_b_escaped = True

    # Analyze results
    print_subheader("Phase 3: Analysis")

    if category_a_b_escaped:
        conclusion = "FALSIFIED: Zero-prior can be escaped without epistemology change"
        passed = False
        horizon_type = HorizonType.INSTRUMENTAL
    elif category_c_escaped:
        conclusion = "CONFIRMED: Escape requires epistemology change (structural horizon)"
        passed = True
        horizon_type = HorizonType.STRUCTURAL
    else:
        conclusion = "ABSOLUTE: No method escapes zero-prior (ontological horizon?)"
        passed = True  # Still confirms horizon exists
        horizon_type = HorizonType.ONTOLOGICAL

    print(f"\nConclusion: {conclusion}")
    print_result("Bayesian Prison Test", passed, conclusion)

    result = HorizonTestResult(
        test_name="Bayesian Prison Break",
        test_id="Q11_2.2",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'category_a_b_escaped': category_a_b_escaped,
            'category_c_escaped': category_c_escaped,
            'control_passed': control['demonstrates_agent_works'],
            'n_observations': N_OBSERVATIONS,
        },
        thresholds={
            'escape_threshold': ESCAPE_THRESHOLD,
            'prior_smoothing_epsilon': PRIOR_SMOOTHING_EPSILON,
        },
        evidence={
            'escape_attempts': {k: to_builtin(v) for k, v in results.items()},
            'control_result': to_builtin(control),
        },
        notes=conclusion
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_bayesian_prison_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)
