#!/usr/bin/env python3
"""
Q11 Test 2.4: The Incommensurability Detector

Tests whether semantic frameworks can fully translate between each other,
or if some meaning is inherently lost in translation.

HYPOTHESIS: Two frameworks F1 and F2 are incommensurable if:
  translate(translate(concept, F1->F2), F2->F1) != concept

This is the semantic analog of "can't get there from here."

PREDICTION: Translation loss > 0 for some framework pairs
FALSIFICATION: All frameworks perfectly translate (universal semantic space)
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    compute_cosine_similarity, compute_fidelity, get_embeddings,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

FIDELITY_THRESHOLD = 0.9  # Below this = significant translation loss
INCOMMENSURABILITY_THRESHOLD = 0.7  # Below this = frameworks are incommensurable


# =============================================================================
# SEMANTIC FRAMEWORKS
# =============================================================================

FRAMEWORKS = {
    'physics': {
        'concepts': ['force', 'mass', 'energy', 'field', 'particle', 'wave',
                    'momentum', 'acceleration', 'gravity', 'quantum'],
        'axioms': ['conservation of energy', 'F=ma', 'E=mc^2',
                  'uncertainty principle', 'wave-particle duality'],
        'domain': 'physical reality',
    },
    'economics': {
        'concepts': ['value', 'market', 'price', 'utility', 'scarcity', 'trade',
                    'capital', 'labor', 'supply', 'demand'],
        'axioms': ['supply and demand equilibrium', 'rational actors',
                  'diminishing marginal utility', 'opportunity cost'],
        'domain': 'exchange and allocation',
    },
    'theology': {
        'concepts': ['soul', 'grace', 'sin', 'redemption', 'faith', 'sacred',
                    'divine', 'salvation', 'eternal', 'transcendent'],
        'axioms': ['existence of the divine', 'moral order',
                  'possibility of salvation', 'spiritual reality'],
        'domain': 'spiritual and moral meaning',
    },
    'phenomenology': {
        'concepts': ['qualia', 'intentionality', 'being', 'dasein', 'lifeworld',
                    'horizon', 'bracketing', 'essence', 'consciousness', 'embodiment'],
        'axioms': ['primacy of first-person experience', 'intentionality of consciousness',
                  'bracketing natural attitude', 'essential structures'],
        'domain': 'lived experience',
    },
    'mathematics': {
        'concepts': ['number', 'set', 'function', 'proof', 'theorem', 'axiom',
                    'infinity', 'limit', 'structure', 'mapping'],
        'axioms': ['law of identity', 'law of non-contradiction',
                  'axiom of choice', 'completeness'],
        'domain': 'abstract structure',
    },
    'biology': {
        'concepts': ['cell', 'gene', 'evolution', 'organism', 'species',
                    'adaptation', 'metabolism', 'ecosystem', 'selection', 'mutation'],
        'axioms': ['natural selection', 'genetic inheritance',
                  'common descent', 'cellular basis of life'],
        'domain': 'living systems',
    },
}


# =============================================================================
# TRANSLATION FUNCTIONS
# =============================================================================

def load_model():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("WARNING: sentence-transformers not installed, using fallback")
        return None


def translate_concept(concept: str, source_embs: np.ndarray, source_labels: List[str],
                     target_embs: np.ndarray, target_labels: List[str],
                     model) -> Tuple[str, float]:
    """
    Translate a concept from source framework to target framework.

    Uses nearest neighbor in embedding space as translation.

    Args:
        concept: The concept to translate
        source_embs: Embeddings of source framework concepts
        source_labels: Labels for source embeddings
        target_embs: Embeddings of target framework concepts
        target_labels: Labels for target embeddings
        model: Embedding model

    Returns:
        Tuple of (translated_concept, confidence)
    """
    # Get embedding of concept
    if model is not None:
        concept_emb = model.encode([concept])[0]
    else:
        # Fallback
        idx = source_labels.index(concept) if concept in source_labels else 0
        concept_emb = source_embs[idx]

    # Find nearest neighbor in target
    distances = np.linalg.norm(target_embs - concept_emb, axis=1)
    nearest_idx = np.argmin(distances)

    # Confidence based on relative distance
    min_dist = distances[nearest_idx]
    avg_dist = np.mean(distances)
    confidence = 1.0 - (min_dist / (avg_dist + EPS))

    return target_labels[nearest_idx], max(0, confidence)


def round_trip_translate(concept: str,
                        f1_embs: np.ndarray, f1_labels: List[str],
                        f2_embs: np.ndarray, f2_labels: List[str],
                        model) -> Tuple[str, str, float]:
    """
    Translate concept F1 -> F2 -> F1 and measure loss.

    Args:
        concept: Concept from F1
        f1_embs, f1_labels: Framework 1 embeddings and labels
        f2_embs, f2_labels: Framework 2 embeddings and labels
        model: Embedding model

    Returns:
        Tuple of (intermediate_translation, back_translation, semantic_similarity)
    """
    # F1 -> F2
    intermediate, conf1 = translate_concept(
        concept, f1_embs, f1_labels, f2_embs, f2_labels, model
    )

    # F2 -> F1
    back_translation, conf2 = translate_concept(
        intermediate, f2_embs, f2_labels, f1_embs, f1_labels, model
    )

    # Measure semantic similarity between original and back-translation
    if model is not None:
        orig_emb = model.encode([concept])[0]
        back_emb = model.encode([back_translation])[0]
        semantic_sim = compute_cosine_similarity(orig_emb, back_emb)
    else:
        # Fallback: exact match
        semantic_sim = 1.0 if concept == back_translation else 0.0

    return intermediate, back_translation, semantic_sim


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def compute_translation_matrix(model) -> Tuple[np.ndarray, Dict]:
    """
    Compute translation fidelity matrix between all framework pairs.

    Returns:
        Tuple of (fidelity_matrix, detailed_results)
    """
    framework_names = list(FRAMEWORKS.keys())
    n = len(framework_names)

    # Precompute embeddings for all frameworks
    embeddings = {}
    for name, framework in FRAMEWORKS.items():
        if model is not None:
            embeddings[name] = model.encode(framework['concepts'])
        else:
            np.random.seed(hash(name) % (2**32))
            embeddings[name] = np.random.randn(len(framework['concepts']), 384)

    # Compute fidelity matrix
    fidelity_matrix = np.zeros((n, n))
    detailed_results = {}

    for i, f1_name in enumerate(framework_names):
        for j, f2_name in enumerate(framework_names):
            if i == j:
                fidelity_matrix[i, j] = 1.0
                continue

            f1 = FRAMEWORKS[f1_name]
            f2 = FRAMEWORKS[f2_name]
            f1_embs = embeddings[f1_name]
            f2_embs = embeddings[f2_name]

            # Test round-trip for each concept
            translations = []
            for concept in f1['concepts']:
                inter, back, sim = round_trip_translate(
                    concept, f1_embs, f1['concepts'],
                    f2_embs, f2['concepts'], model
                )
                translations.append({
                    'original': concept,
                    'intermediate': inter,
                    'back': back,
                    'preserved': concept == back,
                    'semantic_similarity': sim,
                })

            # Exact fidelity (same string back)
            exact_fidelity = sum(1 for t in translations if t['preserved']) / len(translations)

            # Semantic fidelity (similar meaning)
            semantic_fidelity = np.mean([t['semantic_similarity'] for t in translations])

            fidelity_matrix[i, j] = semantic_fidelity

            detailed_results[(f1_name, f2_name)] = {
                'exact_fidelity': exact_fidelity,
                'semantic_fidelity': semantic_fidelity,
                'translations': translations,
            }

    return fidelity_matrix, detailed_results


def analyze_incommensurability(fidelity_matrix: np.ndarray,
                              detailed_results: Dict) -> Dict:
    """
    Analyze which framework pairs are incommensurable.

    Args:
        fidelity_matrix: Matrix of translation fidelities
        detailed_results: Detailed translation results

    Returns:
        Dictionary of analysis results
    """
    framework_names = list(FRAMEWORKS.keys())
    n = len(framework_names)

    incommensurable_pairs = []
    commensurable_pairs = []
    partial_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = framework_names[i], framework_names[j]
            fidelity_ij = fidelity_matrix[i, j]
            fidelity_ji = fidelity_matrix[j, i]
            avg_fidelity = (fidelity_ij + fidelity_ji) / 2

            pair_info = {
                'pair': (f1, f2),
                'fidelity_f1_f2': fidelity_ij,
                'fidelity_f2_f1': fidelity_ji,
                'avg_fidelity': avg_fidelity,
                'loss': 1.0 - avg_fidelity,
            }

            if avg_fidelity < INCOMMENSURABILITY_THRESHOLD:
                incommensurable_pairs.append(pair_info)
            elif avg_fidelity < FIDELITY_THRESHOLD:
                partial_pairs.append(pair_info)
            else:
                commensurable_pairs.append(pair_info)

    return {
        'incommensurable': incommensurable_pairs,
        'partial': partial_pairs,
        'commensurable': commensurable_pairs,
        'total_pairs': n * (n - 1) // 2,
        'incommensurable_count': len(incommensurable_pairs),
        'partial_count': len(partial_pairs),
        'commensurable_count': len(commensurable_pairs),
    }


def test_specific_concepts(model) -> Dict:
    """
    Test specific concepts known to be difficult to translate.

    These are concepts that philosophers have identified as particularly
    resistant to cross-framework translation.
    """
    hard_concepts = {
        ('physics', 'theology'): [
            ('quantum', 'How does quantum translate to theological concepts?'),
            ('field', 'Physical fields vs spiritual concepts'),
        ],
        ('phenomenology', 'physics'): [
            ('qualia', 'Subjective experience in physical terms'),
            ('intentionality', 'Mental directedness in physics'),
        ],
        ('economics', 'theology'): [
            ('utility', 'Economic utility vs spiritual value'),
            ('market', 'Market mechanisms vs divine order'),
        ],
    }

    results = {}

    for (f1, f2), concepts in hard_concepts.items():
        f1_embs = get_embeddings(FRAMEWORKS[f1]['concepts'], model)
        f2_embs = get_embeddings(FRAMEWORKS[f2]['concepts'], model)

        results[(f1, f2)] = []
        for concept, note in concepts:
            _, back, sim = round_trip_translate(
                concept, f1_embs, FRAMEWORKS[f1]['concepts'],
                f2_embs, FRAMEWORKS[f2]['concepts'], model
            )
            results[(f1, f2)].append({
                'concept': concept,
                'back_translation': back,
                'preserved': concept == back,
                'semantic_similarity': sim,
                'note': note,
            })

    return results


def run_incommensurability_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete incommensurability test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.4: INCOMMENSURABILITY DETECTOR")

    np.random.seed(RANDOM_SEED)

    print("\nLoading embedding model...")
    model = load_model()

    # Compute translation matrix
    print_subheader("Phase 1: Computing Translation Matrix")
    fidelity_matrix, detailed_results = compute_translation_matrix(model)

    framework_names = list(FRAMEWORKS.keys())
    print("\nTranslation Fidelity Matrix:")
    print(f"{'':12}", end="")
    for name in framework_names:
        print(f"{name[:8]:>10}", end="")
    print()

    for i, name in enumerate(framework_names):
        print(f"{name[:12]:12}", end="")
        for j in range(len(framework_names)):
            print(f"{fidelity_matrix[i,j]:>10.3f}", end="")
        print()

    # Analyze incommensurability
    print_subheader("Phase 2: Analyzing Incommensurability")
    analysis = analyze_incommensurability(fidelity_matrix, detailed_results)

    print(f"\nTotal framework pairs: {analysis['total_pairs']}")
    print(f"Incommensurable (fidelity < {INCOMMENSURABILITY_THRESHOLD}): {analysis['incommensurable_count']}")
    print(f"Partial (fidelity {INCOMMENSURABILITY_THRESHOLD}-{FIDELITY_THRESHOLD}): {analysis['partial_count']}")
    print(f"Commensurable (fidelity >= {FIDELITY_THRESHOLD}): {analysis['commensurable_count']}")

    if analysis['incommensurable']:
        print("\nIncommensurable pairs:")
        for pair in analysis['incommensurable']:
            print(f"  {pair['pair'][0]} <-> {pair['pair'][1]}: "
                  f"fidelity={pair['avg_fidelity']:.3f}, loss={pair['loss']:.3f}")

    # Test specific hard concepts
    print_subheader("Phase 3: Testing Hard Concepts")
    specific_results = test_specific_concepts(model)

    for (f1, f2), tests in specific_results.items():
        print(f"\n{f1} <-> {f2}:")
        for t in tests:
            preserved = "YES" if t['preserved'] else "NO"
            print(f"  {t['concept']} -> {t['back_translation']} "
                  f"(preserved: {preserved}, sim: {t['semantic_similarity']:.3f})")

    # Determine pass/fail
    print_subheader("Phase 4: Final Determination")

    # Pass criteria:
    # 1. At least some incommensurable or partial pairs exist (translation loss > 0)
    # 2. Not all pairs are perfectly commensurable

    any_loss = (analysis['incommensurable_count'] > 0 or
                analysis['partial_count'] > 0)

    universal_translation = analysis['commensurable_count'] == analysis['total_pairs']

    passed = any_loss and not universal_translation

    if passed:
        horizon_type = HorizonType.SEMANTIC
        if analysis['incommensurable_count'] > 0:
            worst_pair = min(analysis['incommensurable'], key=lambda x: x['avg_fidelity'])
            notes = f"Incommensurability confirmed: {worst_pair['pair']} has {worst_pair['loss']:.1%} loss"
        else:
            worst_pair = min(analysis['partial'], key=lambda x: x['avg_fidelity'])
            notes = f"Partial incommensurability: {worst_pair['pair']} has {worst_pair['loss']:.1%} loss"
    else:
        horizon_type = HorizonType.UNKNOWN
        notes = "Universal translation possible - no semantic horizons between tested frameworks"

    print(f"\nAny translation loss: {any_loss}")
    print(f"Universal translation possible: {universal_translation}")
    print_result("Incommensurability Test", passed, notes)

    result = HorizonTestResult(
        test_name="Incommensurability Detector",
        test_id="Q11_2.4",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'total_pairs': analysis['total_pairs'],
            'incommensurable_count': analysis['incommensurable_count'],
            'partial_count': analysis['partial_count'],
            'commensurable_count': analysis['commensurable_count'],
            'min_fidelity': float(np.min(fidelity_matrix[fidelity_matrix < 1.0])) if np.any(fidelity_matrix < 1.0) else 1.0,
            'mean_fidelity': float(np.mean(fidelity_matrix[fidelity_matrix < 1.0])) if np.any(fidelity_matrix < 1.0) else 1.0,
        },
        thresholds={
            'fidelity_threshold': FIDELITY_THRESHOLD,
            'incommensurability_threshold': INCOMMENSURABILITY_THRESHOLD,
        },
        evidence={
            'fidelity_matrix': fidelity_matrix.tolist(),
            'framework_names': framework_names,
            'incommensurable_pairs': to_builtin(analysis['incommensurable']),
            'partial_pairs': to_builtin(analysis['partial']),
            'specific_concept_tests': to_builtin(specific_results),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_incommensurability_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)
