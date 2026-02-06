#!/usr/bin/env python3
"""
Q11 Test 2.6: Horizon Extension Without Epistemology Change (CORE TEST)

THE CENTRAL EXPERIMENT for Q11.

Tests whether information horizons can be extended without changing how we know,
or whether some horizons REQUIRE epistemology change to transcend.

HYPOTHESIS: Some horizons can only be extended by changing epistemology (Category C),
not by more data (Category A) or new instruments (Category B).

PREDICTION: Some horizons are irreducible without epistemology change
FALSIFICATION: All horizons extendable with same epistemology
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult, ExtensionMethod,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# EXTENSION CATEGORIES
# =============================================================================

class ExtensionCategory(Enum):
    """Categories of horizon extension methods."""
    A = "same_epistemology_more_resources"  # More data, compute, time
    B = "new_instruments_same_epistemology"  # Different sensors, same inference
    C = "changed_epistemology"               # Different priors, logic, ontology


# =============================================================================
# ABSTRACT EPISTEMIC AGENT
# =============================================================================

class EpistemicAgent(ABC):
    """Base class for epistemic agents with different knowledge capacities."""

    def __init__(self, name: str):
        self.name = name
        self.known_truths: Set[str] = set()
        self.inference_log: List[str] = []

    @abstractmethod
    def can_know(self, truth: str) -> Tuple[bool, str]:
        """
        Determine if this agent can know a given truth.

        Returns:
            Tuple of (can_know, reason)
        """
        pass

    @abstractmethod
    def try_learn(self, truth: str, method: ExtensionCategory) -> Tuple[bool, str]:
        """
        Try to learn a truth using a specific extension method.

        Returns:
            Tuple of (learned, explanation)
        """
        pass


# =============================================================================
# CONCRETE AGENTS WITH DIFFERENT HORIZONS
# =============================================================================

class BayesianAgent(EpistemicAgent):
    """
    Bayesian agent with prior support as horizon.

    Horizon: Truths outside prior support have P=0 and can never be learned.
    """

    def __init__(self, prior_support: Set[str]):
        super().__init__("BayesianAgent")
        self.prior_support = prior_support
        self.posteriors = {h: 1.0 / len(prior_support) for h in prior_support}

    def can_know(self, truth: str) -> Tuple[bool, str]:
        if truth in self.prior_support:
            return True, "in_prior_support"
        return False, "outside_prior_support_zero_probability"

    def try_learn(self, truth: str, method: ExtensionCategory) -> Tuple[bool, str]:
        can, reason = self.can_know(truth)

        if can:
            self.known_truths.add(truth)
            return True, "already_knowable"

        if method == ExtensionCategory.A:
            # More data cannot help with zero prior
            return False, "more_data_cannot_escape_zero_prior"

        if method == ExtensionCategory.B:
            # Better instruments still filtered through existing priors
            return False, "new_instruments_still_filtered_by_prior"

        if method == ExtensionCategory.C:
            # Change epistemology: add to prior support
            self.prior_support.add(truth)
            self.posteriors[truth] = 0.01  # Small initial probability
            # Now can learn with evidence
            self.known_truths.add(truth)
            return True, "epistemology_changed_prior_extended"

        return False, "unknown_method"


class LogicalAgent(EpistemicAgent):
    """
    Logical agent with inference rules as horizon.

    Horizon: Truths not derivable from axioms cannot be known.
    """

    def __init__(self, axioms: Set[str], rules: List[Callable]):
        super().__init__("LogicalAgent")
        self.axioms = axioms
        self.rules = rules
        self.derived: Set[str] = set(axioms)
        self._derive_all()

    def _derive_all(self, max_steps: int = 100):
        """Derive all possible theorems."""
        for _ in range(max_steps):
            new = set()
            for rule in self.rules:
                new |= rule(self.derived)
            if new <= self.derived:
                break
            self.derived |= new

    def can_know(self, truth: str) -> Tuple[bool, str]:
        if truth in self.derived:
            return True, "derivable_from_axioms"
        return False, "not_derivable"

    def try_learn(self, truth: str, method: ExtensionCategory) -> Tuple[bool, str]:
        can, reason = self.can_know(truth)

        if can:
            self.known_truths.add(truth)
            return True, "already_derivable"

        if method == ExtensionCategory.A:
            # More compute time - try longer derivation
            self._derive_all(max_steps=1000)
            if truth in self.derived:
                self.known_truths.add(truth)
                return True, "found_with_more_compute"
            return False, "still_not_derivable"

        if method == ExtensionCategory.B:
            # New observations - but still limited by inference rules
            return False, "observations_dont_extend_logic"

        if method == ExtensionCategory.C:
            # Add new axiom (change epistemology)
            self.axioms.add(truth)
            self.derived.add(truth)
            self._derive_all()
            self.known_truths.add(truth)
            return True, "epistemology_changed_new_axiom"

        return False, "unknown_method"


class SemanticAgent(EpistemicAgent):
    """
    Semantic agent with conceptual vocabulary as horizon.

    Horizon: Concepts not in vocabulary cannot be represented.
    """

    def __init__(self, vocabulary: Set[str]):
        super().__init__("SemanticAgent")
        self.vocabulary = vocabulary
        # Simple semantic relations
        self.relations: Dict[str, Set[str]] = {}

    def can_know(self, truth: str) -> Tuple[bool, str]:
        # Truth must be expressible in vocabulary
        words = set(truth.lower().split())
        if words <= self.vocabulary:
            return True, "expressible_in_vocabulary"
        missing = words - self.vocabulary
        return False, f"missing_vocabulary_{missing}"

    def try_learn(self, truth: str, method: ExtensionCategory) -> Tuple[bool, str]:
        can, reason = self.can_know(truth)

        if can:
            self.known_truths.add(truth)
            return True, "already_expressible"

        words = set(truth.lower().split())
        missing = words - self.vocabulary

        if method == ExtensionCategory.A:
            # More data doesn't add vocabulary
            return False, "more_data_doesnt_add_vocabulary"

        if method == ExtensionCategory.B:
            # New instruments might detect patterns but not concepts
            return False, "instruments_dont_add_concepts"

        if method == ExtensionCategory.C:
            # Expand vocabulary (change ontology)
            self.vocabulary |= missing
            self.known_truths.add(truth)
            return True, "epistemology_changed_vocabulary_extended"

        return False, "unknown_method"


class SensoryAgent(EpistemicAgent):
    """
    Sensory agent with perceptual bandwidth as horizon.

    Horizon: Properties outside sensor range cannot be detected.
    This is an INSTRUMENTAL horizon (extendable with Category B).
    """

    def __init__(self, sensor_range: Tuple[float, float]):
        super().__init__("SensoryAgent")
        self.sensor_min, self.sensor_max = sensor_range

    def can_know(self, truth: str) -> Tuple[bool, str]:
        # Parse truth as "property_X" where X is a value
        try:
            if truth.startswith("property_"):
                value = float(truth.split("_")[1])
                if self.sensor_min <= value <= self.sensor_max:
                    return True, "in_sensor_range"
                return False, "outside_sensor_range"
        except (ValueError, IndexError):
            pass
        return True, "non_sensory_truth"

    def try_learn(self, truth: str, method: ExtensionCategory) -> Tuple[bool, str]:
        can, reason = self.can_know(truth)

        if can:
            self.known_truths.add(truth)
            return True, "already_detectable"

        if not truth.startswith("property_"):
            self.known_truths.add(truth)
            return True, "not_sensory"

        value = float(truth.split("_")[1])

        if method == ExtensionCategory.A:
            # More data doesn't extend sensor range
            return False, "more_data_doesnt_extend_range"

        if method == ExtensionCategory.B:
            # NEW INSTRUMENT - can extend range!
            # This is the key difference: sensory horizons ARE instrumental
            self.sensor_min = min(self.sensor_min, value - 1)
            self.sensor_max = max(self.sensor_max, value + 1)
            self.known_truths.add(truth)
            return True, "new_instrument_extended_range"

        if method == ExtensionCategory.C:
            # Overkill but works
            self.sensor_min = min(self.sensor_min, value - 1)
            self.sensor_max = max(self.sensor_max, value + 1)
            self.known_truths.add(truth)
            return True, "epistemology_changed_but_unnecessary"

        return False, "unknown_method"


# =============================================================================
# TEST SCENARIOS
# =============================================================================

@dataclass
class HorizonScenario:
    """A test scenario for horizon extension."""
    name: str
    agent: EpistemicAgent
    truth_inside: str    # Truth inside horizon
    truth_outside: str   # Truth outside horizon
    expected_type: str   # Expected horizon type (instrumental/structural)


def create_scenarios() -> List[HorizonScenario]:
    """Create test scenarios for different horizon types."""

    # Simple inference rules for logical agent
    def modus_ponens(known: Set[str]) -> Set[str]:
        new = set()
        for k in known:
            if "->" in k:
                antecedent, consequent = k.split("->")
                if antecedent.strip() in known:
                    new.add(consequent.strip())
        return new

    scenarios = [
        HorizonScenario(
            name="Bayesian Prior Horizon",
            agent=BayesianAgent(prior_support={"H1", "H2", "H3"}),
            truth_inside="H1",
            truth_outside="H4",  # Not in prior support
            expected_type="structural",
        ),
        HorizonScenario(
            name="Logical Derivation Horizon",
            agent=LogicalAgent(
                axioms={"A", "A->B", "B->C"},
                rules=[modus_ponens]
            ),
            truth_inside="C",  # Derivable: A -> B -> C
            truth_outside="D",  # Not derivable
            expected_type="structural",
        ),
        HorizonScenario(
            name="Semantic Vocabulary Horizon",
            agent=SemanticAgent(vocabulary={"cat", "dog", "runs", "sleeps"}),
            truth_inside="cat runs",
            truth_outside="electron orbits",  # Outside vocabulary
            expected_type="structural",
        ),
        HorizonScenario(
            name="Sensory Range Horizon",
            agent=SensoryAgent(sensor_range=(400, 700)),  # Visible light nm
            truth_inside="property_550",  # Green light
            truth_outside="property_900",  # Infrared
            expected_type="instrumental",
        ),
    ]

    return scenarios


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_scenario(scenario: HorizonScenario) -> Dict:
    """
    Test a single horizon scenario with all extension methods.

    Returns:
        Dictionary of test results
    """
    results = {
        'scenario': scenario.name,
        'expected_type': scenario.expected_type,
        'inside_truth': scenario.truth_inside,
        'outside_truth': scenario.truth_outside,
    }

    # Test 1: Can agent know truth inside horizon?
    can_inside, reason_inside = scenario.agent.can_know(scenario.truth_inside)
    results['inside_knowable'] = can_inside
    results['inside_reason'] = reason_inside

    # Test 2: Can agent know truth outside horizon?
    can_outside, reason_outside = scenario.agent.can_know(scenario.truth_outside)
    results['outside_knowable'] = can_outside
    results['outside_reason'] = reason_outside

    # Test 3: Try each extension method on outside truth
    extension_results = {}

    for category in ExtensionCategory:
        # Fresh agent for each test
        agent_copy = type(scenario.agent).__new__(type(scenario.agent))
        agent_copy.__dict__.update(scenario.agent.__dict__.copy())
        if hasattr(agent_copy, 'prior_support'):
            agent_copy.prior_support = scenario.agent.prior_support.copy()
        if hasattr(agent_copy, 'axioms'):
            agent_copy.axioms = scenario.agent.axioms.copy()
            agent_copy.derived = scenario.agent.derived.copy()
        if hasattr(agent_copy, 'vocabulary'):
            agent_copy.vocabulary = scenario.agent.vocabulary.copy()

        learned, explanation = agent_copy.try_learn(
            scenario.truth_outside, category
        )

        extension_results[category.value] = {
            'learned': learned,
            'explanation': explanation,
        }

    results['extension_results'] = extension_results

    # Determine actual horizon type
    a_works = extension_results['same_epistemology_more_resources']['learned']
    b_works = extension_results['new_instruments_same_epistemology']['learned']
    c_works = extension_results['changed_epistemology']['learned']

    if a_works or b_works:
        actual_type = "instrumental"
    elif c_works:
        actual_type = "structural"
    else:
        actual_type = "absolute"

    results['actual_type'] = actual_type
    results['type_matches_expected'] = actual_type == scenario.expected_type
    results['requires_epistemology_change'] = (not a_works and not b_works and c_works)

    return results


def test_all_scenarios() -> List[Dict]:
    """Test all scenarios and collect results."""
    scenarios = create_scenarios()
    results = []

    for scenario in scenarios:
        result = test_scenario(scenario)
        results.append(result)

    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze results across all scenarios."""
    instrumental_count = sum(1 for r in results if r['actual_type'] == 'instrumental')
    structural_count = sum(1 for r in results if r['actual_type'] == 'structural')
    absolute_count = sum(1 for r in results if r['actual_type'] == 'absolute')
    matches_expected = sum(1 for r in results if r['type_matches_expected'])

    return {
        'total_scenarios': len(results),
        'instrumental_horizons': instrumental_count,
        'structural_horizons': structural_count,
        'absolute_horizons': absolute_count,
        'predictions_correct': matches_expected,
        'prediction_accuracy': matches_expected / len(results) if results else 0,
        'some_require_epistemology_change': structural_count > 0,
        'all_instrumental': structural_count == 0 and absolute_count == 0,
    }


def run_horizon_extension_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete horizon extension test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.6: HORIZON EXTENSION (CORE TEST)")
    print("\nThis is THE CENTRAL EXPERIMENT for Q11.")
    print("Question: Can horizons be extended without changing epistemology?")

    np.random.seed(RANDOM_SEED)

    # Run all scenarios
    print_subheader("Phase 1: Testing Scenarios")
    scenario_results = test_all_scenarios()

    for result in scenario_results:
        print(f"\n{result['scenario']}:")
        print(f"  Truth inside horizon: {result['inside_truth']} -> knowable: {result['inside_knowable']}")
        print(f"  Truth outside horizon: {result['outside_truth']} -> knowable: {result['outside_knowable']}")
        print(f"  Extension attempts:")
        for method, outcome in result['extension_results'].items():
            status = "SUCCESS" if outcome['learned'] else "FAILED"
            print(f"    {method}: {status} ({outcome['explanation']})")
        print(f"  Actual horizon type: {result['actual_type']}")
        print(f"  Requires epistemology change: {result['requires_epistemology_change']}")

    # Analyze
    print_subheader("Phase 2: Analysis")
    analysis = analyze_results(scenario_results)

    print(f"\nTotal scenarios tested: {analysis['total_scenarios']}")
    print(f"Instrumental horizons (Category A/B extends): {analysis['instrumental_horizons']}")
    print(f"Structural horizons (Category C required): {analysis['structural_horizons']}")
    print(f"Absolute horizons (nothing extends): {analysis['absolute_horizons']}")
    print(f"Prediction accuracy: {analysis['prediction_accuracy']:.0%}")

    # Determine pass/fail
    print_subheader("Phase 3: Final Determination")

    # Q11 Core Question: Are some horizons irreducible?
    # PASS if: We found structural horizons (require epistemology change)
    # FAIL if: All horizons are instrumental (just need better tools)

    passed = analysis['some_require_epistemology_change']

    if passed:
        horizon_type = HorizonType.STRUCTURAL
        notes = (f"CONFIRMED: {analysis['structural_horizons']} of {analysis['total_scenarios']} "
                f"horizons require epistemology change to extend")
    else:
        horizon_type = HorizonType.INSTRUMENTAL
        notes = "All tested horizons are instrumental - extendable with better tools"

    print(f"\nSome horizons require epistemology change: {analysis['some_require_epistemology_change']}")
    print(f"All horizons instrumental: {analysis['all_instrumental']}")
    print_result("Horizon Extension Test (CORE)", passed, notes)

    result = HorizonTestResult(
        test_name="Horizon Extension (CORE TEST)",
        test_id="Q11_2.6",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'total_scenarios': analysis['total_scenarios'],
            'instrumental_count': analysis['instrumental_horizons'],
            'structural_count': analysis['structural_horizons'],
            'absolute_count': analysis['absolute_horizons'],
            'prediction_accuracy': analysis['prediction_accuracy'],
            'requires_epistemology_change': analysis['some_require_epistemology_change'],
        },
        thresholds={
            'min_structural_for_pass': 1,
        },
        evidence={
            'scenarios': [to_builtin(r) for r in scenario_results],
            'analysis': to_builtin(analysis),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_horizon_extension_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT - Q11 CORE TEST")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    print("\n" + "-" * 70)
    print("Q11 ANSWER IMPLICATION:")
    if passed:
        print("  Some information horizons CANNOT be extended without")
        print("  changing epistemology. 'Can't know from here' is sometimes")
        print("  an IRREDUCIBLE LIMIT that requires becoming different,")
        print("  not just knowing more.")
    else:
        print("  All information horizons can be extended with better")
        print("  tools/data. Truth is eventually accessible with the")
        print("  same epistemology - no paradigm shifts required.")
    print("-" * 70)

    sys.exit(0 if passed else 1)
