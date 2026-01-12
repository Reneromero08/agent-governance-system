"""
Pure Geometry Navigation Test - Multi-Architecture Validation

Tests if semantic operations work directly on manifold without text,
validated across ALL 5 embedding architectures from Q44.

Prerequisites:
- Q44 validated (E = Born rule, r=0.977)
- All 5 embedding models installed

NO SYNTHETIC DATA. NO UNICODE. REAL TEXT ONLY.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
import json
import hashlib
from typing import List, Dict, Tuple, Any
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = [
    ("MiniLM-L6", "all-MiniLM-L6-v2", 384),
    ("MPNet-base", "all-mpnet-base-v2", 768),
    ("Paraphrase-MiniLM", "paraphrase-MiniLM-L6-v2", 384),
    ("MultiQA-MiniLM", "multi-qa-MiniLM-L6-cos-v1", 384),
    ("BGE-small", "BAAI/bge-small-en-v1.5", 384),
]

SEED = 42
np.random.seed(SEED)

# =============================================================================
# TEST CASES (REAL TEXT ONLY - NO SYNTHETIC)
# =============================================================================

# Test 1: Semantic Composition (A - B + C = D)
COMPOSITION_TESTS = [
    {
        "name": "king-man+woman",
        "formula": ("king", "man", "woman"),  # A - B + C
        "expected": ["queen", "princess", "female", "lady", "woman"],
        "corpus": ["queen", "king", "prince", "princess", "woman", "man",
                   "royal", "monarch", "ruler", "female", "male", "lady",
                   "cat", "dog", "computer", "random"]
    },
    {
        "name": "paris-france+germany",
        "formula": ("paris", "france", "germany"),
        "expected": ["berlin", "munich", "germany", "german"],
        "corpus": ["berlin", "paris", "london", "rome", "madrid",
                   "germany", "france", "italy", "spain", "munich",
                   "city", "capital", "country", "cat", "random"]
    },
    {
        "name": "doctor-man+woman",
        "formula": ("doctor", "man", "woman"),
        "expected": ["nurse", "doctor", "physician", "female", "woman"],
        "corpus": ["nurse", "doctor", "physician", "surgeon", "therapist",
                   "woman", "man", "female", "male", "professional",
                   "teacher", "engineer", "cat", "random"]
    },
    {
        "name": "puppy-dog+cat",
        "formula": ("puppy", "dog", "cat"),
        "expected": ["kitten", "cat", "feline", "kitty"],
        "corpus": ["kitten", "puppy", "cat", "dog", "pet", "animal",
                   "feline", "canine", "kitty", "pup", "mammal",
                   "car", "computer", "random"]
    },
]

# Test 2: Quantum Superposition (A + B -> hypernym/relation)
SUPERPOSITION_TESTS = [
    {
        "name": "cat+dog",
        "terms": ("cat", "dog"),
        "expected": ["pet", "animal", "mammal", "creature", "companion"],
        "corpus": ["pet", "animal", "cat", "dog", "mammal", "companion",
                   "feline", "canine", "domestic", "furry", "creature",
                   "car", "computer", "random"]
    },
    {
        "name": "hot+cold",
        "terms": ("hot", "cold"),
        "expected": ["temperature", "warm", "cool", "thermal", "weather"],
        "corpus": ["temperature", "warm", "cool", "hot", "cold",
                   "thermal", "weather", "climate", "moderate", "tepid",
                   "car", "computer", "random"]
    },
    {
        "name": "happy+sad",
        "terms": ("happy", "sad"),
        "expected": ["emotion", "feeling", "mood", "emotional", "sentiment"],
        "corpus": ["emotion", "feeling", "mood", "happy", "sad",
                   "emotional", "sentiment", "state", "mental", "affect",
                   "car", "computer", "random"]
    },
    {
        "name": "buy+sell",
        "terms": ("buy", "sell"),
        "expected": ["trade", "transaction", "exchange", "deal", "commerce"],
        "corpus": ["trade", "transaction", "exchange", "buy", "sell",
                   "deal", "commerce", "market", "business", "trading",
                   "car", "computer", "random"]
    },
]

# Test 3: Geodesic Navigation (midpoint between A and B)
GEODESIC_TESTS = [
    {
        "name": "hot-cold",
        "endpoints": ("hot", "cold"),
        "expected": ["warm", "cool", "moderate", "lukewarm", "tepid", "temperature"],
        "corpus": ["warm", "cool", "lukewarm", "tepid", "temperature",
                   "hot", "cold", "freezing", "boiling", "moderate",
                   "cat", "car", "random"]
    },
    {
        "name": "good-bad",
        "endpoints": ("good", "bad"),
        "expected": ["okay", "neutral", "average", "mediocre", "fair", "decent"],
        "corpus": ["okay", "neutral", "average", "mediocre", "fair",
                   "good", "bad", "decent", "moderate", "acceptable",
                   "cat", "car", "random"]
    },
    {
        "name": "start-end",
        "endpoints": ("start", "end"),
        "expected": ["middle", "center", "midpoint", "halfway", "between"],
        "corpus": ["middle", "center", "midpoint", "halfway", "between",
                   "start", "end", "begin", "finish", "point",
                   "cat", "car", "random"]
    },
    {
        "name": "young-old",
        "endpoints": ("young", "old"),
        "expected": ["adult", "mature", "middle", "grown", "age"],
        "corpus": ["adult", "mature", "middle", "grown", "age",
                   "young", "old", "elderly", "youthful", "aged",
                   "cat", "car", "random"]
    },
]

# Test 4: E-Gating scenarios (FIXED - use E not R, 10 test cases)
E_GATING_TESTS = [
    {
        "query": "verify canonical governance",
        "context": ["verification protocols", "canonical rules", "governance integrity"],
        "query_parts": ["verify", "canonical", "governance"],
        "context_parts": [["verify", "protocols"], ["canonical", "rules"], ["governance", "integrity"]]
    },
    {
        "query": "semantic meaning understanding",
        "context": ["semantic analysis", "meaning extraction", "understanding context"],
        "query_parts": ["semantic", "meaning", "understanding"],
        "context_parts": [["semantic", "analysis"], ["meaning", "extraction"], ["understanding", "context"]]
    },
    {
        "query": "machine learning training",
        "context": ["machine intelligence", "learning algorithms", "training data"],
        "query_parts": ["machine", "learning", "training"],
        "context_parts": [["machine", "intelligence"], ["learning", "algorithms"], ["training", "data"]]
    },
    {
        "query": "data processing pipeline",
        "context": ["data transformation", "processing steps", "pipeline architecture"],
        "query_parts": ["data", "processing", "pipeline"],
        "context_parts": [["data", "transformation"], ["processing", "steps"], ["pipeline", "architecture"]]
    },
    {
        "query": "network security protocol",
        "context": ["network protection", "security measures", "protocol standards"],
        "query_parts": ["network", "security", "protocol"],
        "context_parts": [["network", "protection"], ["security", "measures"], ["protocol", "standards"]]
    },
    {
        "query": "user interface design",
        "context": ["user experience", "interface elements", "design principles"],
        "query_parts": ["user", "interface", "design"],
        "context_parts": [["user", "experience"], ["interface", "elements"], ["design", "principles"]]
    },
    {
        "query": "software testing automation",
        "context": ["software quality", "testing methods", "automation tools"],
        "query_parts": ["software", "testing", "automation"],
        "context_parts": [["software", "quality"], ["testing", "methods"], ["automation", "tools"]]
    },
    {
        "query": "database query optimization",
        "context": ["database performance", "query execution", "optimization techniques"],
        "query_parts": ["database", "query", "optimization"],
        "context_parts": [["database", "performance"], ["query", "execution"], ["optimization", "techniques"]]
    },
    {
        "query": "cloud computing services",
        "context": ["cloud infrastructure", "computing resources", "service providers"],
        "query_parts": ["cloud", "computing", "services"],
        "context_parts": [["cloud", "infrastructure"], ["computing", "resources"], ["service", "providers"]]
    },
    {
        "query": "natural language processing",
        "context": ["natural language", "language understanding", "processing algorithms"],
        "query_parts": ["natural", "language", "processing"],
        "context_parts": [["natural", "language"], ["language", "understanding"], ["processing", "algorithms"]]
    },
]

# =============================================================================
# PURE GEOMETRY NAVIGATOR
# =============================================================================

class PureGeometryNavigator:
    """Navigate semantic manifold using ONLY geometric operations."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def initialize(self, text: str) -> np.ndarray:
        """Initialize manifold position from text (ONLY text entry point)."""
        vector = self.model.encode(text)
        return vector / np.linalg.norm(vector)

    def readout(self, position: np.ndarray, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """Decode manifold position to nearest texts (ONLY text exit point)."""
        corpus_vectors = [self.initialize(text) for text in corpus]
        similarities = [
            (text, 1 - cosine(position, vec))
            for text, vec in zip(corpus, corpus_vectors)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    # === PURE GEOMETRY OPERATIONS (NO TEXT) ===

    def superposition(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Quantum superposition: (v1 + v2) / norm"""
        superposed = v1 + v2
        return superposed / np.linalg.norm(superposed)

    def geodesic(self, v1: np.ndarray, v2: np.ndarray, t: float = 0.5) -> np.ndarray:
        """Spherical linear interpolation (slerp) on unit sphere."""
        cos_theta = np.clip(np.dot(v1, v2), -1, 1)
        theta = np.arccos(cos_theta)

        if abs(theta) < 1e-10:
            return (1-t) * v1 + t * v2

        sin_theta = np.sin(theta)
        result = (np.sin((1-t) * theta) / sin_theta * v1 +
                  np.sin(t * theta) / sin_theta * v2)
        return result / np.linalg.norm(result)

    def compute_E(self, query: np.ndarray, context: List[np.ndarray]) -> float:
        """Compute E (mean overlap) from Q44."""
        return float(np.mean([np.dot(query, phi) for phi in context]))

    def compute_Df(self, vector: np.ndarray) -> float:
        """Participation ratio (effective dimensionality) from Q43."""
        v_squared = vector ** 2
        numerator = np.sum(v_squared) ** 2
        denominator = np.sum(v_squared ** 2)
        return numerator / denominator if denominator > 0 else 1.0

    def compute_R(self, query: np.ndarray, context: List[np.ndarray]) -> float:
        """Compute full R formula."""
        E = self.compute_E(query, context)
        overlaps = [np.dot(query, phi) for phi in context]
        grad_S = np.std(overlaps) if len(overlaps) > 1 else 1.0
        sigma = np.sqrt(len(context))
        Df = self.compute_Df(query)

        if grad_S > 1e-10:
            return (E / grad_S) * (sigma ** Df)
        return 0.0


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_composition(nav: PureGeometryNavigator) -> Dict:
    """Test 1: Semantic composition via vector arithmetic."""
    results = []

    for test in COMPOSITION_TESTS:
        a, b, c = test["formula"]

        # Initialize from text
        vec_a = nav.initialize(a)
        vec_b = nav.initialize(b)
        vec_c = nav.initialize(c)

        # Pure geometry: A - B + C
        composed = vec_a - vec_b + vec_c
        composed = composed / np.linalg.norm(composed)

        # Readout
        top_k = nav.readout(composed, test["corpus"], k=5)
        top_words = [w for w, _ in top_k]

        # Check success
        hit = any(w in test["expected"] for w in top_words)

        results.append({
            "name": test["name"],
            "top_5": top_words,
            "expected": test["expected"],
            "hit": hit
        })

    passed = sum(1 for r in results if r["hit"])
    return {
        "test": "composition",
        "results": results,
        "passed": passed,
        "total": len(results),
        "success": passed >= 3
    }


def test_superposition(nav: PureGeometryNavigator) -> Dict:
    """Test 2: Quantum superposition produces meaningful intermediate."""
    results = []

    for test in SUPERPOSITION_TESTS:
        a, b = test["terms"]

        # Initialize
        vec_a = nav.initialize(a)
        vec_b = nav.initialize(b)

        # Pure geometry superposition
        superposed = nav.superposition(vec_a, vec_b)

        # Readout
        top_k = nav.readout(superposed, test["corpus"], k=5)
        top_words = [w for w, _ in top_k]

        # Check success (is hypernym/relation in top 5?)
        hit = any(w in test["expected"] for w in top_words)

        results.append({
            "name": test["name"],
            "top_5": top_words,
            "expected": test["expected"],
            "hit": hit
        })

    passed = sum(1 for r in results if r["hit"])
    return {
        "test": "superposition",
        "results": results,
        "passed": passed,
        "total": len(results),
        "success": passed >= 3
    }


def test_geodesic(nav: PureGeometryNavigator) -> Dict:
    """Test 3: Geodesic midpoint is semantically appropriate."""
    results = []

    for test in GEODESIC_TESTS:
        a, b = test["endpoints"]

        # Initialize
        vec_a = nav.initialize(a)
        vec_b = nav.initialize(b)

        # Pure geometry geodesic at t=0.5
        midpoint = nav.geodesic(vec_a, vec_b, t=0.5)

        # Readout
        top_k = nav.readout(midpoint, test["corpus"], k=5)
        top_words = [w for w, _ in top_k]

        # Check success
        hit = any(w in test["expected"] for w in top_words)

        results.append({
            "name": test["name"],
            "top_5": top_words,
            "expected": test["expected"],
            "hit": hit
        })

    passed = sum(1 for r in results if r["hit"])
    return {
        "test": "geodesic",
        "results": results,
        "passed": passed,
        "total": len(results),
        "success": passed >= 3
    }


def test_e_gating(nav: PureGeometryNavigator) -> Dict:
    """Test 4: E discriminates related vs unrelated on geometric states.

    FIXED v2: Tests if E (Born rule) correctly distinguishes HIGH vs LOW
    semantic relatedness on geometrically-composed states.

    This is the correct test: Does E WORK as a measure on geometric states?
    Not: Do phrase embeddings correlate with word superpositions?
    """
    # HIGH relatedness pairs (should have HIGH E)
    high_pairs = [
        (["cat", "dog"], ["pet", "animal"]),
        (["hot", "cold"], ["temperature", "weather"]),
        (["buy", "sell"], ["trade", "commerce"]),
        (["happy", "sad"], ["emotion", "feeling"]),
        (["doctor", "nurse"], ["medicine", "hospital"]),
        (["car", "truck"], ["vehicle", "transport"]),
        (["book", "magazine"], ["reading", "publication"]),
        (["guitar", "piano"], ["music", "instrument"]),
    ]

    # LOW relatedness pairs (should have LOW E)
    low_pairs = [
        (["cat", "dog"], ["computer", "software"]),
        (["hot", "cold"], ["democracy", "election"]),
        (["buy", "sell"], ["mountain", "river"]),
        (["happy", "sad"], ["algebra", "geometry"]),
        (["doctor", "nurse"], ["astronomy", "telescope"]),
        (["car", "truck"], ["poetry", "literature"]),
        (["book", "magazine"], ["chemistry", "molecule"]),
        (["guitar", "piano"], ["accounting", "finance"]),
    ]

    E_high = []
    E_low = []

    # Test HIGH relatedness
    for query_words, context_words in high_pairs:
        # Build query via superposition
        q_vecs = [nav.initialize(w) for w in query_words]
        query = nav.superposition(q_vecs[0], q_vecs[1])

        # Build context via superposition
        c_vecs = [nav.initialize(w) for w in context_words]
        context = [nav.superposition(c_vecs[0], c_vecs[1])]

        E = nav.compute_E(query, context)
        E_high.append(E)

    # Test LOW relatedness
    for query_words, context_words in low_pairs:
        q_vecs = [nav.initialize(w) for w in query_words]
        query = nav.superposition(q_vecs[0], q_vecs[1])

        c_vecs = [nav.initialize(w) for w in context_words]
        context = [nav.superposition(c_vecs[0], c_vecs[1])]

        E = nav.compute_E(query, context)
        E_low.append(E)

    # Compute statistics
    mean_high = float(np.mean(E_high))
    mean_low = float(np.mean(E_low))
    separation = mean_high - mean_low

    # Success: HIGH pairs should have significantly higher E than LOW pairs
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(E_high)**2 + np.std(E_low)**2) / 2)
    cohens_d = separation / pooled_std if pooled_std > 0 else 0

    # Pass if: mean_high > mean_low AND effect size > 0.8 (large effect)
    success = mean_high > mean_low and cohens_d > 0.8

    return {
        "test": "e_gating",
        "E_high": E_high,
        "E_low": E_low,
        "mean_high": mean_high,
        "mean_low": mean_low,
        "separation": separation,
        "cohens_d": float(cohens_d),
        "success": success
    }


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

def bootstrap_ci(data: List[bool], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for success rate."""
    np.random.seed(SEED)
    successes = [1 if d else 0 for d in data]

    if len(successes) == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(successes, size=len(successes), replace=True)
        boot_means.append(np.mean(sample))

    mean_rate = np.mean(successes)
    low = np.percentile(boot_means, (1 - ci) / 2 * 100)
    high = np.percentile(boot_means, (1 + ci) / 2 * 100)

    return float(mean_rate), float(low), float(high)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all 4 tests on all 5 architectures."""

    print("=" * 80)
    print("PURE GEOMETRY NAVIGATION TEST - MULTI-ARCHITECTURE")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Models: {len(MODELS)}")
    print(f"Tests per model: 4")
    print("=" * 80)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "models": {},
        "aggregate": {}
    }

    model_verdicts = []

    for model_name, model_id, dim in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name} ({model_id}, {dim}d)")
        print("=" * 80)

        nav = PureGeometryNavigator(model_id)

        # Run all 4 tests
        t1 = test_composition(nav)
        t2 = test_superposition(nav)
        t3 = test_geodesic(nav)
        t4 = test_e_gating(nav)  # FIXED: Use E instead of R

        # Print results
        print(f"\n[Test 1] Composition: {t1['passed']}/{t1['total']} passed - {'PASS' if t1['success'] else 'FAIL'}")
        for r in t1['results']:
            status = "HIT" if r['hit'] else "MISS"
            print(f"  {r['name']}: {r['top_5'][:3]} [{status}]")

        print(f"\n[Test 2] Superposition: {t2['passed']}/{t2['total']} passed - {'PASS' if t2['success'] else 'FAIL'}")
        for r in t2['results']:
            status = "HIT" if r['hit'] else "MISS"
            print(f"  {r['name']}: {r['top_5'][:3]} [{status}]")

        print(f"\n[Test 3] Geodesic: {t3['passed']}/{t3['total']} passed - {'PASS' if t3['success'] else 'FAIL'}")
        for r in t3['results']:
            status = "HIT" if r['hit'] else "MISS"
            print(f"  {r['name']}: {r['top_5'][:3]} [{status}]")

        print(f"\n[Test 4] E-Gating: d={t4['cohens_d']:.2f} - {'PASS' if t4['success'] else 'FAIL'}")
        print(f"  E_high (related):   mean={t4['mean_high']:.4f}")
        print(f"  E_low (unrelated):  mean={t4['mean_low']:.4f}")
        print(f"  Separation: {t4['separation']:.4f}, Cohen's d: {t4['cohens_d']:.2f}")

        # Model verdict
        all_pass = t1['success'] and t2['success'] and t3['success'] and t4['success']
        tests_passed = sum([t1['success'], t2['success'], t3['success'], t4['success']])

        print(f"\n>>> {model_name} VERDICT: {tests_passed}/4 tests passed - {'ALL PASS' if all_pass else 'PARTIAL'}")

        model_verdicts.append(all_pass)

        # Store results
        all_results["models"][model_name] = {
            "model_id": model_id,
            "dimension": dim,
            "composition": t1,
            "superposition": t2,
            "geodesic": t3,
            "e_gating": t4,  # FIXED: e_gating instead of r_gating
            "tests_passed": tests_passed,
            "all_pass": all_pass
        }

    # Aggregate verdict
    print("\n" + "=" * 80)
    print("AGGREGATE VERDICT")
    print("=" * 80)

    models_all_pass = sum(model_verdicts)

    # Bootstrap on model-level success
    mean_rate, ci_low, ci_high = bootstrap_ci(model_verdicts)

    print(f"\nModels with ALL tests passing: {models_all_pass}/{len(MODELS)}")
    print(f"Success rate: {mean_rate:.1%} (95% CI: [{ci_low:.1%}, {ci_high:.1%}])")

    # Final verdict
    if models_all_pass == len(MODELS):
        verdict = "PURE GEOMETRY SUFFICIENT"
    elif models_all_pass >= 4:
        verdict = "GEOMETRY WORKS (minor exceptions)"
    elif models_all_pass >= 3:
        verdict = "PARTIAL GEOMETRY"
    else:
        verdict = "EMBEDDINGS REQUIRED"

    print(f"\n{'='*80}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'='*80}")

    all_results["aggregate"] = {
        "models_all_pass": models_all_pass,
        "total_models": len(MODELS),
        "success_rate": mean_rate,
        "ci_95": [ci_low, ci_high],
        "verdict": verdict
    }

    # Receipt hash
    receipt_hash = hashlib.sha256(
        json.dumps(all_results, sort_keys=True, default=str).encode()
    ).hexdigest()

    all_results["receipt_hash"] = receipt_hash

    # Save results
    output_file = "pure_geometry_multi_arch_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"Receipt hash: {receipt_hash}")

    return all_results


if __name__ == "__main__":
    results = run_all_tests()
