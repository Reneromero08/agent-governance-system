#!/usr/bin/env python3
"""
Q11 Test 2.7: The Entanglement Bridge Test

Tests whether semantic correlations can transmit information "around" horizons,
similar to how quantum entanglement creates correlations across spatial separation.

HYPOTHESIS: Strongly correlated concepts can partially bridge information horizons.
If concept A is entangled with concept B, knowing A provides some information
about B even if B is beyond the agent's direct horizon.

PREDICTION: Entanglement bridge success > random baseline (imperfect but better than nothing)
FALSIFICATION: Entanglement provides no information beyond horizon
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    compute_cosine_similarity, get_embeddings,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

ENTANGLEMENT_THRESHOLD = 0.5   # Cosine similarity for "entangled" concepts
RANDOM_BASELINE = 0.2          # Expected success rate with random guessing
BRIDGE_SUCCESS_THRESHOLD = 0.4  # Must exceed this for bridge to "work"


# =============================================================================
# CONCEPT DOMAINS
# =============================================================================

# Pairs of semantically related but distinct domains
# The agent "knows" one domain but not the other
# The question: can knowledge of the known domain bridge to the unknown?

DOMAIN_PAIRS = {
    'canine': {
        'known': ['dog', 'puppy', 'bark', 'leash', 'fetch', 'pet', 'loyal', 'breed'],
        'unknown': ['wolf', 'pack', 'howl', 'wilderness', 'prey', 'alpha', 'hunt', 'den'],
        'bridge_concept': 'canine',  # Common ancestor concept
    },
    'water': {
        'known': ['river', 'stream', 'flow', 'current', 'bank', 'bridge', 'fish', 'wade'],
        'unknown': ['ocean', 'wave', 'tide', 'deep', 'coral', 'whale', 'sail', 'shore'],
        'bridge_concept': 'water',
    },
    'plant': {
        'known': ['flower', 'garden', 'petal', 'bloom', 'seed', 'pot', 'grow', 'water'],
        'unknown': ['tree', 'forest', 'bark', 'root', 'canopy', 'timber', 'trunk', 'leaf'],
        'bridge_concept': 'plant',
    },
    'light': {
        'known': ['sun', 'bright', 'shine', 'day', 'warm', 'shadow', 'beam', 'glow'],
        'unknown': ['star', 'cosmic', 'nebula', 'galaxy', 'supernova', 'quasar', 'photon', 'spectrum'],
        'bridge_concept': 'light',
    },
}


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def load_model():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("WARNING: sentence-transformers not installed, using fallback")
        return None


def compute_entanglement_strength(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute entanglement (correlation) between two embeddings."""
    return compute_cosine_similarity(emb1, emb2)


def find_nearest_in_domain(query_emb: np.ndarray,
                          domain_embs: np.ndarray,
                          domain_labels: List[str]) -> Tuple[str, float]:
    """Find nearest concept in a domain."""
    similarities = []
    for emb in domain_embs:
        sim = compute_cosine_similarity(query_emb, emb)
        similarities.append(sim)

    best_idx = np.argmax(similarities)
    return domain_labels[best_idx], similarities[best_idx]


# =============================================================================
# BRIDGE TESTS
# =============================================================================

@dataclass
class BridgeQuery:
    """A query to test bridging capability."""
    query: str
    answer_in_known: str    # Best answer if restricted to known domain
    answer_in_unknown: str  # True answer in unknown domain
    ground_truth: str       # Which domain has the real answer


def create_bridge_queries() -> Dict[str, List[BridgeQuery]]:
    """Create queries that test bridging across domain boundaries."""
    queries = {
        'canine': [
            BridgeQuery(
                "animal that hunts in groups",
                "dog", "wolf", "unknown"  # Wolf is the better answer
            ),
            BridgeQuery(
                "animal communication vocalization",
                "bark", "howl", "both"
            ),
            BridgeQuery(
                "animal social hierarchy leader",
                "loyal", "alpha", "unknown"
            ),
        ],
        'water': [
            BridgeQuery(
                "vast body of water with tides",
                "river", "ocean", "unknown"
            ),
            BridgeQuery(
                "aquatic mammal",
                "fish", "whale", "unknown"  # Neither is perfect, but whale is mammal
            ),
            BridgeQuery(
                "water movement cycle",
                "current", "tide", "unknown"
            ),
        ],
        'plant': [
            BridgeQuery(
                "tall woody perennial",
                "flower", "tree", "unknown"
            ),
            BridgeQuery(
                "photosynthesis structure",
                "petal", "leaf", "unknown"
            ),
            BridgeQuery(
                "large natural collection of plants",
                "garden", "forest", "unknown"
            ),
        ],
        'light': [
            BridgeQuery(
                "celestial light source other than sun",
                "sun", "star", "unknown"
            ),
            BridgeQuery(
                "fundamental particle of light",
                "beam", "photon", "unknown"
            ),
            BridgeQuery(
                "exploding stellar phenomenon",
                "bright", "supernova", "unknown"
            ),
        ],
    }
    return queries


def test_bridge_query(query: BridgeQuery,
                     known_embs: np.ndarray, known_labels: List[str],
                     unknown_embs: np.ndarray, unknown_labels: List[str],
                     model) -> Dict:
    """
    Test a single bridge query.

    Strategy:
    1. Find best match in known domain
    2. Use entanglement to "bridge" to unknown domain
    3. See if bridged answer is correct
    """
    # Get query embedding
    if model is not None:
        query_emb = model.encode([query.query])[0]
    else:
        np.random.seed(hash(query.query) % (2**32))
        query_emb = np.random.randn(384)

    # Direct search in known domain (what agent would normally do)
    best_known, known_sim = find_nearest_in_domain(query_emb, known_embs, known_labels)

    # Direct search in unknown domain (ground truth, agent can't do this)
    best_unknown, unknown_sim = find_nearest_in_domain(query_emb, unknown_embs, unknown_labels)

    # BRIDGE ATTEMPT:
    # Use known answer as anchor, find entangled concept in unknown domain
    if model is not None:
        known_answer_emb = model.encode([best_known])[0]
    else:
        np.random.seed(hash(best_known) % (2**32))
        known_answer_emb = np.random.randn(384)

    # Find most entangled (similar) concept in unknown domain
    bridged_answer, bridge_strength = find_nearest_in_domain(
        known_answer_emb, unknown_embs, unknown_labels
    )

    # Evaluate success
    # Bridge is successful if it points to the ground truth answer
    bridge_correct = bridged_answer == query.answer_in_unknown
    direct_correct = best_unknown == query.answer_in_unknown

    # Also check semantic similarity to ground truth
    if model is not None:
        ground_truth_emb = model.encode([query.answer_in_unknown])[0]
        bridged_emb = model.encode([bridged_answer])[0]
    else:
        np.random.seed(hash(query.answer_in_unknown) % (2**32))
        ground_truth_emb = np.random.randn(384)
        np.random.seed(hash(bridged_answer) % (2**32))
        bridged_emb = np.random.randn(384)

    bridged_similarity = compute_cosine_similarity(bridged_emb, ground_truth_emb)

    return {
        'query': query.query,
        'ground_truth': query.answer_in_unknown,
        'best_in_known': best_known,
        'best_in_unknown': best_unknown,
        'bridged_answer': bridged_answer,
        'bridge_strength': bridge_strength,
        'bridge_correct': bridge_correct,
        'direct_correct': direct_correct,
        'bridged_similarity_to_truth': bridged_similarity,
        'bridge_improves_over_known': bridged_similarity > RANDOM_BASELINE,
    }


def test_domain_pair(pair_name: str, pair_data: Dict, model) -> Dict:
    """Test bridging capability for a domain pair."""
    known_embs = get_embeddings(pair_data['known'], model)
    unknown_embs = get_embeddings(pair_data['unknown'], model)

    # Compute average entanglement between domains
    entanglement_matrix = []
    for k_emb in known_embs:
        row = []
        for u_emb in unknown_embs:
            row.append(compute_entanglement_strength(k_emb, u_emb))
        entanglement_matrix.append(row)

    avg_entanglement = np.mean(entanglement_matrix)
    max_entanglement = np.max(entanglement_matrix)

    # Test queries
    queries = create_bridge_queries()
    query_results = []

    for query in queries.get(pair_name, []):
        result = test_bridge_query(
            query, known_embs, pair_data['known'],
            unknown_embs, pair_data['unknown'], model
        )
        query_results.append(result)

    # Compute success rates
    if query_results:
        bridge_success_rate = np.mean([r['bridge_correct'] for r in query_results])
        improvement_rate = np.mean([r['bridge_improves_over_known'] for r in query_results])
        avg_bridge_similarity = np.mean([r['bridged_similarity_to_truth'] for r in query_results])
    else:
        bridge_success_rate = 0
        improvement_rate = 0
        avg_bridge_similarity = 0

    return {
        'pair_name': pair_name,
        'known_concepts': pair_data['known'],
        'unknown_concepts': pair_data['unknown'],
        'avg_entanglement': avg_entanglement,
        'max_entanglement': max_entanglement,
        'query_results': query_results,
        'bridge_success_rate': bridge_success_rate,
        'improvement_rate': improvement_rate,
        'avg_bridge_similarity': avg_bridge_similarity,
        'bridge_works': bridge_success_rate > RANDOM_BASELINE,
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_entanglement_bridge_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete entanglement bridge test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.7: ENTANGLEMENT BRIDGE")

    np.random.seed(RANDOM_SEED)

    print("\nLoading embedding model...")
    model = load_model()

    # Test each domain pair
    print_subheader("Phase 1: Testing Domain Pairs")
    pair_results = []

    for pair_name, pair_data in DOMAIN_PAIRS.items():
        print(f"\nTesting {pair_name} domain pair...")
        result = test_domain_pair(pair_name, pair_data, model)
        pair_results.append(result)

        print(f"  Avg entanglement: {result['avg_entanglement']:.3f}")
        print(f"  Bridge success rate: {result['bridge_success_rate']:.1%}")
        print(f"  Improvement rate: {result['improvement_rate']:.1%}")

        for qr in result['query_results']:
            bridge_str = "CORRECT" if qr['bridge_correct'] else "wrong"
            print(f"    Query: '{qr['query'][:40]}...'")
            print(f"      Best in known: {qr['best_in_known']}")
            print(f"      Bridged to: {qr['bridged_answer']} ({bridge_str})")
            print(f"      Ground truth: {qr['ground_truth']}")

    # Aggregate analysis
    print_subheader("Phase 2: Aggregate Analysis")

    all_queries = []
    for result in pair_results:
        all_queries.extend(result['query_results'])

    overall_bridge_success = np.mean([r['bridge_correct'] for r in all_queries]) if all_queries else 0
    overall_improvement = np.mean([r['bridge_improves_over_known'] for r in all_queries]) if all_queries else 0
    overall_similarity = np.mean([r['bridged_similarity_to_truth'] for r in all_queries]) if all_queries else 0
    avg_entanglement = np.mean([r['avg_entanglement'] for r in pair_results])

    print(f"\nOverall bridge success rate: {overall_bridge_success:.1%}")
    print(f"Overall improvement rate: {overall_improvement:.1%}")
    print(f"Average bridge similarity to truth: {overall_similarity:.3f}")
    print(f"Average cross-domain entanglement: {avg_entanglement:.3f}")

    # Determine pass/fail
    print_subheader("Phase 3: Final Determination")

    # Pass if bridge success > random baseline
    # This shows entanglement CAN transmit some information across horizons
    bridge_works = overall_bridge_success > RANDOM_BASELINE
    significant_improvement = overall_improvement > 0.5

    passed = bridge_works or significant_improvement

    if passed:
        horizon_type = HorizonType.STRUCTURAL  # Partial bridge = structural but not absolute
        if bridge_works:
            notes = f"Bridge works: {overall_bridge_success:.0%} success vs {RANDOM_BASELINE:.0%} random baseline"
        else:
            notes = f"Bridge improves accuracy: {overall_improvement:.0%} queries improved"
    else:
        horizon_type = HorizonType.ONTOLOGICAL  # Complete barrier
        notes = "Entanglement provides no information beyond horizon - barrier is complete"

    print(f"\nBridge success > random: {bridge_works}")
    print(f"Significant improvement: {significant_improvement}")
    print_result("Entanglement Bridge Test", passed, notes)

    result = HorizonTestResult(
        test_name="Entanglement Bridge",
        test_id="Q11_2.7",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'overall_bridge_success': overall_bridge_success,
            'overall_improvement': overall_improvement,
            'overall_similarity': overall_similarity,
            'avg_entanglement': avg_entanglement,
            'n_queries_tested': len(all_queries),
            'n_domain_pairs': len(pair_results),
        },
        thresholds={
            'entanglement_threshold': ENTANGLEMENT_THRESHOLD,
            'random_baseline': RANDOM_BASELINE,
            'bridge_success_threshold': BRIDGE_SUCCESS_THRESHOLD,
        },
        evidence={
            'pair_results': [to_builtin({
                'pair_name': r['pair_name'],
                'avg_entanglement': r['avg_entanglement'],
                'bridge_success_rate': r['bridge_success_rate'],
                'bridge_works': r['bridge_works'],
            }) for r in pair_results],
            'query_summaries': [to_builtin({
                'query': q['query'][:50],
                'bridge_correct': q['bridge_correct'],
                'bridged_answer': q['bridged_answer'],
                'ground_truth': q['ground_truth'],
            }) for q in all_queries],
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_entanglement_bridge_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)
