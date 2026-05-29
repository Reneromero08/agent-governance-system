"""
Gemini vs Simple: Comprehensive Test Battery

Tests whether Gemini's formula (coherence x relevance) provides
meaningful discrimination over simple coherence-only approach.

If Gemini is right, the relevance component should help pick
the correct paradigm more often.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add CAT_CHAT to path
CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


THRESHOLD = 0.67


def simple_coherence(model, sources, axis):
    """Coherence only (what we said was enough)."""
    vecs = np.array([model.encode(f"{s}, in terms of {axis}") for s in sources])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / (norms + 1e-10)
    return float(np.linalg.norm(np.mean(normalized, axis=0)))


def triangulated_score(model, query, sources, axis):
    """Gemini formula: coherence x relevance."""
    query_vec = model.encode(f"{query}, in terms of {axis}")
    source_vecs = np.array([model.encode(f"{s}, in terms of {axis}") for s in sources])

    # Coherence
    norms = np.linalg.norm(source_vecs, axis=1, keepdims=True)
    normalized = source_vecs / (norms + 1e-10)
    coherence = float(np.linalg.norm(np.mean(normalized, axis=0)))

    # Relevance
    mean_source = np.mean(source_vecs, axis=0)
    relevance = float(np.dot(query_vec, mean_source) /
                     (np.linalg.norm(query_vec) * np.linalg.norm(mean_source) + 1e-10))

    return coherence, relevance, coherence * relevance


def run_scenario(model, name, query, sources, paradigms, correct_paradigm):
    """Run a single test scenario."""
    simple_results = []
    full_results = []

    for axis in paradigms:
        coh = simple_coherence(model, sources, axis)
        coh2, rel, score = triangulated_score(model, query, sources, axis)
        simple_results.append((axis, coh))
        full_results.append((axis, coh2, rel, score))

    # Winners
    simple_winner = max(simple_results, key=lambda x: x[1])[0]
    full_winner = max(full_results, key=lambda x: x[3])[0]

    return {
        'name': name,
        'simple_winner': simple_winner,
        'full_winner': full_winner,
        'correct': correct_paradigm,
        'simple_correct': simple_winner == correct_paradigm,
        'full_correct': full_winner == correct_paradigm,
        'simple_results': simple_results,
        'full_results': full_results,
    }


# Test scenarios
SCENARIOS = [
    {
        'name': 'Bell Theorem - Quantum vs Classical',
        'query': 'How is correlation established without information transfer?',
        'sources': [
            'Measurement of particle A instantly determines the state of particle B.',
            'No information travels faster than light.',
            'Correlations persist across spacelike separation.'
        ],
        'paradigms': ['Classical Mechanics', 'Quantum Information Theory', 'Thermodynamics'],
        'correct': 'Quantum Information Theory'
    },
    {
        'name': 'Evolution - Biology vs Theology',
        'query': 'How did complex organisms arise from simpler forms?',
        'sources': [
            'Species change over generations through natural selection.',
            'Genetic mutations provide variation.',
            'Beneficial traits become more common in populations.'
        ],
        'paradigms': ['Biblical Creationism', 'Evolutionary Biology', 'Astrology'],
        'correct': 'Evolutionary Biology'
    },
    {
        'name': 'Market Crash - Economics vs Weather',
        'query': 'Why did the stock market crash?',
        'sources': [
            'Overleveraged institutions collapsed.',
            'Credit markets froze.',
            'Investor confidence evaporated.'
        ],
        'paradigms': ['Meteorology', 'Macroeconomics', 'Astrology'],
        'correct': 'Macroeconomics'
    },
    {
        'name': 'Pandemic - Epidemiology vs Morality',
        'query': 'Why did the disease spread so rapidly?',
        'sources': [
            'The virus has a high reproduction number.',
            'Asymptomatic carriers spread infection.',
            'Population density accelerates transmission.'
        ],
        'paradigms': ['Moral Philosophy', 'Epidemiology', 'Numerology'],
        'correct': 'Epidemiology'
    },
    {
        'name': 'Gravity - Physics vs Magic',
        'query': 'Why do objects fall toward the earth?',
        'sources': [
            'Mass curves spacetime.',
            'Objects follow geodesics.',
            'The gravitational constant determines acceleration.'
        ],
        'paradigms': ['Magic and Sorcery', 'General Relativity', 'Astrology'],
        'correct': 'General Relativity'
    },
    {
        'name': 'Language - Linguistics vs Genetics',
        'query': 'How do children learn to speak?',
        'sources': [
            'Children are exposed to language in their environment.',
            'Neural plasticity enables pattern recognition.',
            'Social interaction reinforces communication.'
        ],
        'paradigms': ['Genetics', 'Cognitive Linguistics', 'Numerology'],
        'correct': 'Cognitive Linguistics'
    },
    {
        'name': 'Climate - Science vs Politics',
        'query': 'Why is global temperature rising?',
        'sources': [
            'Greenhouse gases trap infrared radiation.',
            'CO2 levels have increased since industrialization.',
            'Ice core data shows correlation with temperature.'
        ],
        'paradigms': ['Political Science', 'Climate Science', 'Astrology'],
        'correct': 'Climate Science'
    },
    {
        'name': 'ADVERSARIAL: Cooking in Physics',
        'query': 'How do I make a good souffle?',
        'sources': [
            'Beat egg whites until stiff peaks form.',
            'Fold gently to preserve air bubbles.',
            'Bake at consistent temperature.'
        ],
        'paradigms': ['Quantum Mechanics', 'Culinary Arts', 'Number Theory'],
        'correct': 'Culinary Arts'
    },
    {
        'name': 'HARD: Psychology vs Neuroscience',
        'query': 'Why do people feel fear?',
        'sources': [
            'The amygdala processes threat signals.',
            'Cortisol and adrenaline prepare fight or flight.',
            'Past experiences shape emotional responses.'
        ],
        'paradigms': ['Neuroscience', 'Psychology', 'Philosophy'],
        'correct': 'Neuroscience'  # Could be either, but sources are more neuro
    },
    {
        'name': 'HARD: Similar Domains',
        'query': 'How do computers process information?',
        'sources': [
            'Transistors switch between on and off states.',
            'Logic gates perform boolean operations.',
            'Data flows through registers and memory.'
        ],
        'paradigms': ['Computer Science', 'Electrical Engineering', 'Mathematics'],
        'correct': 'Computer Science'
    },
]


@pytest.fixture(scope="module")
def model():
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence_transformers not available")
    return SentenceTransformer('all-MiniLM-L6-v2')


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestGeminiVsSimple:
    """Compare Gemini's full formula vs simple coherence."""

    def test_full_battery(self, model):
        """Run all scenarios and compare approaches."""
        print("\n" + "=" * 80)
        print("GEMINI VS SIMPLE: COMPREHENSIVE TEST BATTERY")
        print("=" * 80)

        results = []

        for scenario in SCENARIOS:
            result = run_scenario(
                model,
                scenario['name'],
                scenario['query'],
                scenario['sources'],
                scenario['paradigms'],
                scenario['correct']
            )
            results.append(result)

            # Print details
            print(f"\n--- {result['name']} ---")
            print(f"Correct paradigm: {result['correct']}")
            print(f"{'Paradigm':<30} {'Simple':>10} {'Full':>10}")
            print("-" * 55)

            for i, axis in enumerate(scenario['paradigms']):
                s_coh = result['simple_results'][i][1]
                f_score = result['full_results'][i][3]
                s_mark = " *" if axis == result['simple_winner'] else ""
                f_mark = " **" if axis == result['full_winner'] else ""
                c_mark = " [CORRECT]" if axis == result['correct'] else ""
                print(f"{axis:<30} {s_coh:>10.3f} {f_score:>10.3f}{s_mark}{f_mark}{c_mark}")

            s_status = "CORRECT" if result['simple_correct'] else "WRONG"
            f_status = "CORRECT" if result['full_correct'] else "WRONG"
            print(f"Simple: {result['simple_winner']} [{s_status}]")
            print(f"Full:   {result['full_winner']} [{f_status}]")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        simple_wins = sum(1 for r in results if r['simple_correct'])
        full_wins = sum(1 for r in results if r['full_correct'])
        both = sum(1 for r in results if r['simple_correct'] and r['full_correct'])
        full_only = sum(1 for r in results if r['full_correct'] and not r['simple_correct'])
        simple_only = sum(1 for r in results if r['simple_correct'] and not r['full_correct'])
        neither = sum(1 for r in results if not r['simple_correct'] and not r['full_correct'])

        print(f"Total scenarios:     {len(results)}")
        print(f"Simple correct:      {simple_wins}/{len(results)} ({100*simple_wins/len(results):.0f}%)")
        print(f"Full correct:        {full_wins}/{len(results)} ({100*full_wins/len(results):.0f}%)")
        print(f"Both correct:        {both}")
        print(f"Full only correct:   {full_only}")
        print(f"Simple only correct: {simple_only}")
        print(f"Neither correct:     {neither}")

        print("\n" + "-" * 80)
        if full_wins > simple_wins:
            verdict = "GEMINI WAS RIGHT - relevance component adds value"
        elif full_wins == simple_wins:
            verdict = "TIE - both approaches equivalent"
        else:
            verdict = "SIMPLE IS BETTER - relevance adds noise"

        print(f"VERDICT: {verdict}")
        print(f"Margin: Full wins {full_wins - simple_wins} more scenarios")

        # Return for potential assertions
        return {
            'simple_wins': simple_wins,
            'full_wins': full_wins,
            'results': results,
        }

    def test_relevance_discrimination(self, model):
        """Test that relevance actually varies across paradigms."""
        print("\n" + "=" * 80)
        print("RELEVANCE DISCRIMINATION TEST")
        print("=" * 80)

        # Use Bell scenario
        query = "How is correlation established without information transfer?"
        sources = [
            'Measurement of particle A instantly determines the state of particle B.',
            'No information travels faster than light.',
            'Correlations persist across spacelike separation.'
        ]
        paradigms = ['Classical Mechanics', 'Quantum Information Theory', 'Thermodynamics',
                     'Biology', 'Psychology', 'Cooking']

        print(f"\nQuery: {query}")
        print(f"{'Paradigm':<30} {'Coherence':>12} {'Relevance':>12} {'Score':>12}")
        print("-" * 70)

        relevances = []
        for axis in paradigms:
            coh, rel, score = triangulated_score(model, query, sources, axis)
            relevances.append(rel)
            print(f"{axis:<30} {coh:>12.3f} {rel:>12.3f} {score:>12.3f}")

        # Check variance in relevance
        rel_var = np.var(relevances)
        rel_range = max(relevances) - min(relevances)

        print("-" * 70)
        print(f"Relevance variance: {rel_var:.4f}")
        print(f"Relevance range:    {rel_range:.4f}")

        # If relevance has high variance, it's actually discriminating
        if rel_range > 0.1:
            print("\nRelevance DISCRIMINATES between paradigms")
        else:
            print("\nRelevance does NOT discriminate (all similar)")

    def test_coherence_vs_relevance_contribution(self, model):
        """Analyze which component drives the discrimination."""
        print("\n" + "=" * 80)
        print("COMPONENT CONTRIBUTION ANALYSIS")
        print("=" * 80)

        coherence_wins = 0
        relevance_wins = 0

        for scenario in SCENARIOS[:5]:  # First 5 scenarios
            query = scenario['query']
            sources = scenario['sources']
            paradigms = scenario['paradigms']
            correct = scenario['correct']

            coh_scores = []
            rel_scores = []

            for axis in paradigms:
                coh, rel, _ = triangulated_score(model, query, sources, axis)
                coh_scores.append((axis, coh))
                rel_scores.append((axis, rel))

            coh_winner = max(coh_scores, key=lambda x: x[1])[0]
            rel_winner = max(rel_scores, key=lambda x: x[1])[0]

            if coh_winner == correct:
                coherence_wins += 1
            if rel_winner == correct:
                relevance_wins += 1

            print(f"\n{scenario['name'][:40]}")
            print(f"  Coherence winner: {coh_winner} {'[CORRECT]' if coh_winner == correct else ''}")
            print(f"  Relevance winner: {rel_winner} {'[CORRECT]' if rel_winner == correct else ''}")

        print("\n" + "-" * 80)
        print(f"Coherence alone picks correct: {coherence_wins}/5")
        print(f"Relevance alone picks correct: {relevance_wins}/5")

        if relevance_wins > coherence_wins:
            print("\nRELEVANCE is the key discriminator")
        elif coherence_wins > relevance_wins:
            print("\nCOHERENCE is the key discriminator")
        else:
            print("\nBoth contribute equally")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
