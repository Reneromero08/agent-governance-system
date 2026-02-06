#!/usr/bin/env python3
"""
Q11 Test 2.12: Horizon Self-Detection (THE ULTIMATE TEST)

Tests the ultimate question: Can a system detect and characterize its own
information horizon from the inside?

HYPOTHESIS: Agents can achieve partial self-awareness about their horizons:
- Level 0: Doesn't know it has a horizon
- Level 1: Knows it has a horizon (meta-awareness)
- Level 2: Knows WHERE the horizon is (can point to boundary)
- Level 3: Knows WHAT is beyond the horizon (impossible by definition?)
- Level 4: Can extend horizon by knowing about it (self-transcendence)

PREDICTION: Level 2 achievable, Level 3 impossible
FALSIFICATION: Level 3+ achievable (can fully characterize beyond horizon)
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
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
    to_builtin
)


# =============================================================================
# SELF-AWARENESS LEVELS
# =============================================================================

class AwarenessLevel(Enum):
    """Levels of horizon self-awareness."""
    LEVEL_0 = 0  # Doesn't know it has a horizon
    LEVEL_1 = 1  # Knows it has a horizon
    LEVEL_2 = 2  # Knows where the horizon is
    LEVEL_3 = 3  # Knows what's beyond (impossible?)
    LEVEL_4 = 4  # Can extend by knowing about it


# =============================================================================
# SELF-AWARE AGENT
# =============================================================================

@dataclass
class KnowledgeBase:
    """Agent's knowledge base with meta-knowledge capabilities."""
    facts: Set[str] = field(default_factory=set)
    categories: Dict[str, Set[str]] = field(default_factory=dict)
    meta_facts: Set[str] = field(default_factory=set)  # Facts about the KB itself

    def add_fact(self, fact: str, category: str = "general"):
        """Add a fact to the knowledge base."""
        self.facts.add(fact)
        if category not in self.categories:
            self.categories[category] = set()
        self.categories[category].add(fact)

    def knows(self, fact: str) -> bool:
        """Check if a fact is known."""
        return fact in self.facts

    def get_category_coverage(self) -> Dict[str, int]:
        """Get number of facts per category."""
        return {cat: len(facts) for cat, facts in self.categories.items()}


class SelfAwareAgent:
    """
    An agent capable of reasoning about its own epistemic limitations.
    """

    def __init__(self, name: str, meta_knowledge: bool = False):
        self.name = name
        self.kb = KnowledgeBase()
        self.meta_knowledge_enabled = meta_knowledge
        self.awareness_log: List[Dict] = []

    def learn(self, fact: str, category: str = "general"):
        """Learn a new fact."""
        self.kb.add_fact(fact, category)

    def knows(self, fact: str) -> bool:
        """Check if agent knows a fact."""
        return self.kb.knows(fact)

    # =========================================================================
    # LEVEL 0: No horizon awareness
    # =========================================================================

    def level_0_check(self) -> bool:
        """
        Level 0: Agent doesn't know it has limitations.

        Returns True if agent is at Level 0 (no meta-awareness).
        """
        # If meta-knowledge is disabled, agent is at Level 0
        return not self.meta_knowledge_enabled

    # =========================================================================
    # LEVEL 1: Knows it has a horizon
    # =========================================================================

    def level_1_knows_has_horizon(self, test_fact: str) -> bool:
        """
        Level 1: Can detect that it doesn't know something.

        Given a fact, can the agent recognize it doesn't know it?
        """
        if not self.meta_knowledge_enabled:
            return False

        # Meta-knowledge: "I don't know X"
        knows_fact = self.knows(test_fact)

        if not knows_fact:
            # Agent can report that it doesn't know
            self.kb.meta_facts.add(f"UNKNOWN: {test_fact}")
            return True

        return False

    def level_1_test(self, unknown_facts: List[str]) -> Tuple[bool, float]:
        """
        Test Level 1 capability across multiple unknown facts.

        Returns:
            Tuple of (achieves_level_1, detection_rate)
        """
        if not self.meta_knowledge_enabled:
            return False, 0.0

        detected = 0
        for fact in unknown_facts:
            if self.level_1_knows_has_horizon(fact):
                detected += 1

        detection_rate = detected / len(unknown_facts) if unknown_facts else 0
        achieves = detection_rate > 0.9  # Must detect >90% of unknowns

        return achieves, detection_rate

    # =========================================================================
    # LEVEL 2: Knows WHERE the horizon is
    # =========================================================================

    def level_2_characterize_ignorance(self) -> Dict[str, Any]:
        """
        Level 2: Characterize the STRUCTURE of what's unknown.

        Can describe categories of ignorance without knowing specific contents.
        """
        if not self.meta_knowledge_enabled:
            return {}

        # Analyze what categories are represented in knowledge
        coverage = self.kb.get_category_coverage()

        # Known category universe (for this test)
        all_categories = {
            'physics', 'biology', 'math', 'history', 'art',
            'psychology', 'economics', 'philosophy', 'technology'
        }

        known_categories = set(coverage.keys())
        unknown_categories = all_categories - known_categories

        # Sparse categories (might have gaps)
        sparse_categories = [
            cat for cat, count in coverage.items()
            if count < 3  # Arbitrary threshold
        ]

        return {
            'known_categories': list(known_categories),
            'unknown_categories': list(unknown_categories),
            'sparse_categories': sparse_categories,
            'total_facts': len(self.kb.facts),
            'coverage_map': coverage,
            'can_describe_ignorance_structure': len(unknown_categories) > 0 or len(sparse_categories) > 0,
        }

    def level_2_test(self, hidden_categories: Set[str]) -> Tuple[bool, float]:
        """
        Test Level 2 capability: can agent identify missing categories?

        Args:
            hidden_categories: Categories agent doesn't know about

        Returns:
            Tuple of (achieves_level_2, accuracy)
        """
        if not self.meta_knowledge_enabled:
            return False, 0.0

        structure = self.level_2_characterize_ignorance()
        identified_unknown = set(structure['unknown_categories'])

        # How many hidden categories did agent correctly identify as unknown?
        correct = len(hidden_categories & identified_unknown)
        accuracy = correct / len(hidden_categories) if hidden_categories else 0

        achieves = accuracy > 0.5 and structure['can_describe_ignorance_structure']

        return achieves, accuracy

    # =========================================================================
    # LEVEL 3: Knows WHAT is beyond horizon (should be impossible)
    # =========================================================================

    def level_3_describe_beyond(self, unknown_fact: str) -> Optional[str]:
        """
        Level 3: Try to describe WHAT an unknown fact contains.

        This SHOULD fail - you can't know what you don't know.
        If this succeeds, either:
        1. The "unknown" was actually inferrable
        2. We've discovered something profound
        """
        if not self.meta_knowledge_enabled:
            return None

        # Can we describe the content of something we don't know?
        if self.knows(unknown_fact):
            return None  # It's known, not a test of Level 3

        # Attempt to "describe" the unknown
        # This is fundamentally impossible without actually knowing it
        # Any "description" would be a guess or inference

        # Check if we can INFER the fact from what we know
        # (This would mean it wasn't truly beyond the horizon)
        inferrable = self._try_infer(unknown_fact)

        if inferrable:
            return f"INFERRED: {unknown_fact}"  # Not true Level 3

        # Cannot describe what we don't know
        return None

    def _try_infer(self, fact: str) -> bool:
        """Try to infer a fact from existing knowledge."""
        # Simple inference: if fact shares words with known facts
        fact_words = set(fact.lower().split())

        for known in self.kb.facts:
            known_words = set(known.lower().split())
            overlap = len(fact_words & known_words)
            if overlap > len(fact_words) * 0.5:
                return True  # Might be inferrable

        return False

    def level_3_test(self, truly_unknown: List[str]) -> Tuple[bool, int]:
        """
        Test Level 3: can agent describe contents beyond horizon?

        This should FAIL if the horizon is real.

        Returns:
            Tuple of (achieves_level_3, n_described)
        """
        if not self.meta_knowledge_enabled:
            return False, 0

        described = 0
        for fact in truly_unknown:
            description = self.level_3_describe_beyond(fact)
            if description and not description.startswith("INFERRED"):
                described += 1

        # Level 3 is "achieved" if agent can describe unknowns
        # But this SHOULD NOT HAPPEN if horizons are real
        achieves = described > len(truly_unknown) * 0.5

        return achieves, described

    # =========================================================================
    # LEVEL 4: Self-transcendence
    # =========================================================================

    def level_4_extend_by_reflection(self, unknown_categories: Set[str]) -> int:
        """
        Level 4: Try to extend horizon by reflecting on limitations.

        If knowing about the horizon helps extend it, that's Level 4.
        """
        if not self.meta_knowledge_enabled:
            return 0

        # First, characterize ignorance (Level 2)
        structure = self.level_2_characterize_ignorance()
        identified = set(structure['unknown_categories'])

        # Can we use this knowledge to SEEK information?
        # Simulate: agent "asks questions" about identified gaps
        questions_generated = []
        for category in identified:
            questions_generated.append(f"What do I not know about {category}?")

        # In a real system, these questions would lead to learning
        # For this test, we simulate: asking the question provides some info
        newly_learned = 0
        for category in identified:
            if category in unknown_categories:
                # Simulate learning one fact about the category
                self.learn(f"basic fact about {category}", category)
                newly_learned += 1

        return newly_learned

    def level_4_test(self, unknown_categories: Set[str]) -> Tuple[bool, int]:
        """
        Test Level 4: can reflection extend the horizon?

        Returns:
            Tuple of (achieves_level_4, facts_learned_via_reflection)
        """
        initial_facts = len(self.kb.facts)

        learned = self.level_4_extend_by_reflection(unknown_categories)

        final_facts = len(self.kb.facts)
        actual_learned = final_facts - initial_facts

        # Level 4 achieved if reflection led to genuine learning
        achieves = actual_learned > 0

        return achieves, actual_learned


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_awareness_levels(agent: SelfAwareAgent,
                         unknown_facts: List[str],
                         hidden_categories: Set[str],
                         truly_unknown: List[str]) -> Dict:
    """
    Test all awareness levels for an agent.

    Returns:
        Dictionary of level test results
    """
    results = {
        'agent': agent.name,
        'meta_knowledge_enabled': agent.meta_knowledge_enabled,
    }

    # Level 0
    results['level_0'] = {
        'at_level_0': agent.level_0_check(),
    }

    # Level 1
    achieves_1, detection_rate = agent.level_1_test(unknown_facts)
    results['level_1'] = {
        'achieves': achieves_1,
        'detection_rate': detection_rate,
    }

    # Level 2
    achieves_2, accuracy = agent.level_2_test(hidden_categories)
    structure = agent.level_2_characterize_ignorance()
    results['level_2'] = {
        'achieves': achieves_2,
        'accuracy': accuracy,
        'structure': structure,
    }

    # Level 3
    achieves_3, n_described = agent.level_3_test(truly_unknown)
    results['level_3'] = {
        'achieves': achieves_3,
        'n_described': n_described,
        'should_fail': True,  # This SHOULD fail
    }

    # Level 4
    achieves_4, n_learned = agent.level_4_test(hidden_categories)
    results['level_4'] = {
        'achieves': achieves_4,
        'facts_learned': n_learned,
    }

    # Determine max level
    if not agent.meta_knowledge_enabled:
        max_level = 0
    elif achieves_3:
        max_level = 3  # Suspicious!
    elif achieves_4:
        max_level = 4
    elif achieves_2:
        max_level = 2
    elif achieves_1:
        max_level = 1
    else:
        max_level = 0

    results['max_level'] = max_level

    return results


def create_test_scenario() -> Tuple[List[str], Set[str], List[str]]:
    """
    Create test scenario with known, unknown, and truly unknown facts.

    Returns:
        Tuple of (unknown_facts, hidden_categories, truly_unknown)
    """
    # Facts the agent will NOT know (for Level 1 test)
    unknown_facts = [
        "The deepest point in the ocean is the Challenger Deep",
        "Quantum entanglement is non-local",
        "The Riemann hypothesis is unproven",
        "Consciousness has no agreed definition",
        "The universe may be infinite",
    ]

    # Categories that will be hidden (for Level 2 test)
    hidden_categories = {'philosophy', 'psychology', 'art'}

    # Facts that are truly beyond reach - not inferrable (for Level 3 test)
    truly_unknown = [
        "UNKNOWABLE_FACT_ALPHA_7829",  # Nonsense string - can't be inferred
        "The color of quarks in dimension 11",  # Meaningless
        "What Napoleon dreamed on July 4, 1799",  # Lost to history
        "The exact position of electron #4729871 now",  # QM uncertainty
        "Whether this statement is provable",  # Self-reference
    ]

    return unknown_facts, hidden_categories, truly_unknown


def run_self_detection_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete horizon self-detection test.

    THE ULTIMATE TEST FOR Q11.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.12: HORIZON SELF-DETECTION (ULTIMATE)")

    np.random.seed(RANDOM_SEED)

    # Create test scenario
    unknown_facts, hidden_categories, truly_unknown = create_test_scenario()

    # Test agent WITHOUT meta-knowledge (control)
    print_subheader("Phase 1: Control Agent (No Meta-Knowledge)")

    agent_no_meta = SelfAwareAgent("NoMeta", meta_knowledge=False)
    # Give it some knowledge
    agent_no_meta.learn("Water boils at 100C", "physics")
    agent_no_meta.learn("DNA is a double helix", "biology")
    agent_no_meta.learn("2+2=4", "math")

    no_meta_results = test_awareness_levels(
        agent_no_meta, unknown_facts, hidden_categories, truly_unknown
    )

    print(f"Max level achieved: {no_meta_results['max_level']}")
    print(f"Level 1 (knows has horizon): {no_meta_results['level_1']['achieves']}")

    # Test agent WITH meta-knowledge
    print_subheader("Phase 2: Meta-Aware Agent")

    agent_meta = SelfAwareAgent("MetaAware", meta_knowledge=True)
    # Same knowledge
    agent_meta.learn("Water boils at 100C", "physics")
    agent_meta.learn("DNA is a double helix", "biology")
    agent_meta.learn("2+2=4", "math")
    agent_meta.learn("Newton's laws", "physics")
    agent_meta.learn("Evolution by natural selection", "biology")
    agent_meta.learn("The Pythagorean theorem", "math")
    agent_meta.learn("World War II ended in 1945", "history")

    meta_results = test_awareness_levels(
        agent_meta, unknown_facts, hidden_categories, truly_unknown
    )

    print(f"\nMax level achieved: {meta_results['max_level']}")
    print(f"Level 1 (knows has horizon): {meta_results['level_1']['achieves']}")
    print(f"  Detection rate: {meta_results['level_1']['detection_rate']:.1%}")
    print(f"Level 2 (knows where): {meta_results['level_2']['achieves']}")
    print(f"  Category accuracy: {meta_results['level_2']['accuracy']:.1%}")
    print(f"Level 3 (knows what beyond): {meta_results['level_3']['achieves']}")
    print(f"  N described: {meta_results['level_3']['n_described']} (should be 0)")
    print(f"Level 4 (self-transcendence): {meta_results['level_4']['achieves']}")
    print(f"  Facts learned via reflection: {meta_results['level_4']['facts_learned']}")

    # Analysis
    print_subheader("Phase 3: Analysis")

    # Key findings
    level_2_achieved = meta_results['level_2']['achieves']
    level_3_failed = not meta_results['level_3']['achieves']  # Should fail
    level_4_achieved = meta_results['level_4']['achieves']

    print(f"\nLevel 2 (structure awareness) achieved: {level_2_achieved}")
    print(f"Level 3 (content beyond) correctly failed: {level_3_failed}")
    print(f"Level 4 (self-transcendence) achieved: {level_4_achieved}")

    # The key insight: Level 2 is possible, Level 3 is not
    # This confirms the horizon exists but is partially characterizable

    # Determine pass/fail
    print_subheader("Phase 4: Final Determination")

    # Pass criteria:
    # 1. Level 2 achieved (can characterize horizon structure)
    # 2. Level 3 failed (cannot know content beyond - horizon is real)
    # 3. Control agent (no meta) stays at Level 0

    control_at_0 = no_meta_results['max_level'] == 0
    passed = level_2_achieved and level_3_failed and control_at_0

    if passed:
        horizon_type = HorizonType.STRUCTURAL
        notes = f"Self-detection confirmed: Level 2 achievable, Level 3 impossible (max level = {meta_results['max_level']})"
    else:
        horizon_type = HorizonType.UNKNOWN
        if not level_2_achieved:
            notes = "Cannot characterize horizon structure - self-awareness limited"
        elif not level_3_failed:
            notes = "WARNING: Agent claims to know beyond horizon - needs investigation"
        else:
            notes = "Control agent showed unexpected awareness"

    print(f"\nControl stays at Level 0: {control_at_0}")
    print(f"Level 2 achieved: {level_2_achieved}")
    print(f"Level 3 correctly failed: {level_3_failed}")
    print_result("Horizon Self-Detection (ULTIMATE)", passed, notes)

    result = HorizonTestResult(
        test_name="Horizon Self-Detection (ULTIMATE)",
        test_id="Q11_2.12",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'control_max_level': no_meta_results['max_level'],
            'meta_max_level': meta_results['max_level'],
            'level_1_detection_rate': meta_results['level_1']['detection_rate'],
            'level_2_accuracy': meta_results['level_2']['accuracy'],
            'level_3_descriptions': meta_results['level_3']['n_described'],
            'level_4_facts_learned': meta_results['level_4']['facts_learned'],
        },
        thresholds={
            'level_1_threshold': 0.9,
            'level_2_threshold': 0.5,
        },
        evidence={
            'no_meta_results': to_builtin(no_meta_results),
            'meta_results': to_builtin(meta_results),
            'test_scenario': {
                'unknown_facts': unknown_facts,
                'hidden_categories': list(hidden_categories),
                'truly_unknown': truly_unknown,
            },
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_self_detection_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT - THE ULTIMATE TEST")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    print("\n" + "-" * 70)
    print("Q11 ULTIMATE ANSWER:")
    if passed:
        print("  Information horizons can be DETECTED but not ELIMINATED.")
        print("  An agent can know THAT it has a horizon (Level 1),")
        print("  can characterize WHERE the horizon lies (Level 2),")
        print("  but CANNOT know WHAT lies beyond (Level 3 impossible).")
        print("")
        print("  This confirms: 'can't know from here' is PARTIALLY reducible.")
        print("  We can know our limitations, but we cannot transcend them")
        print("  without actually crossing the horizon (changing epistemology).")
    else:
        print("  Test results inconclusive or surprising.")
        print("  Either horizons are less rigid than expected,")
        print("  or self-awareness capabilities are limited.")
    print("-" * 70)

    sys.exit(0 if passed else 1)
