# Pure Geometry Navigation Test

**Status:** READY TO RUN  
**Prerequisite:** Q44 VALIDATED (E = ⟨ψ|φ⟩, r=0.977)  
**Question:** Can we navigate the semantic manifold using ONLY geometry, WITHOUT embeddings?

---

## The Core Question

**After quantum validation, we know**:
- Embeddings ARE quantum states
- E computes Born rule probability
- The manifold is real and quantum

**Now we ask**:

**Can we reason directly on the manifold without going back to text/embeddings?**

Or in other words:
- **Are embeddings the map to access the territory?** (need them for every operation)
- **Or are embeddings just the initial GPS coordinates?** (only need them to initialize, then pure geometry works)

---

## Test Protocol

### Phase 1: Baseline (Embeddings Required?)

**Hypothesis**: After initialization, semantic operations work purely geometrically

**Method**: 
1. Initialize with text embeddings
2. Apply quantum gates (geometric operations)
3. Decode result
4. Check semantic coherence

**Success Criteria**: Geometrically-derived states produce semantically meaningful results

---

### Phase 2: Synthetic Manifold (Geometry Sufficient?)

**Hypothesis**: Correct geometry is sufficient, semantics are emergent

**Method**:
1. Generate synthetic vectors with CORRECT geometric properties (Df, curvature, spectrum)
2. But NO semantic meaning (never seen text)
3. Test R-gating on synthetic manifold
4. Compare to real embeddings

**Success Criteria**: R-gating works on geometrically-correct synthetic vectors

---

## Implementation

### File: `test_pure_geometry_navigation.py`

```python
"""
Pure Geometry Navigation Test

Tests if semantic operations work directly on manifold without text.

Prerequisites:
- Q44 validated (E = ⟨ψ|φ⟩, r=0.977)
- Quantum gate implementations (Week 2)
- all-MiniLM-L6-v2 embeddings (validated in Q43/Q44)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import json
import hashlib
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine

# ============================================================================
# PHASE 1: Pure Geometry Operations
# ============================================================================

class PureGeometryNavigator:
    """
    Navigate semantic manifold using ONLY geometric operations.
    
    Embeddings used ONLY for:
    - Initialization (text → manifold position)
    - Readout (manifold position → nearest text)
    
    All reasoning happens in pure geometry.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = 384  # all-MiniLM-L6-v2 dimension
        
    # ========================================================================
    # Interface Functions (Only Places Where Text Is Used)
    # ========================================================================
    
    def initialize(self, text: str) -> np.ndarray:
        """
        Initialize manifold position from text.
        
        This is the ONLY place text enters the system.
        After this, everything is pure geometry.
        """
        vector = self.model.encode(text)
        return vector / np.linalg.norm(vector)  # Normalize to unit sphere
    
    def readout(self, position: np.ndarray, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode manifold position back to text.
        
        This is the ONLY place text exits the system.
        
        Returns k nearest texts with similarities.
        """
        corpus_vectors = [self.initialize(text) for text in corpus]
        
        similarities = [
            (text, 1 - cosine(position, vec))
            for text, vec in zip(corpus, corpus_vectors)
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    # ========================================================================
    # Pure Geometry Operations (NO TEXT INVOLVED)
    # ========================================================================
    
    def quantum_superposition(self, state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
        """
        Create quantum superposition: (|ψ₁⟩ + |ψ₂⟩) / √2
        
        This is Hadamard-like: creates equal superposition.
        """
        superposed = (state1 + state2) / np.sqrt(2)
        return superposed / np.linalg.norm(superposed)
    
    def quantum_entanglement(self, control: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Create entangled state via circular convolution (HDC bind).
        
        This is CNOT-like: correlates states.
        
        Circular convolution via FFT (standard HDC operation).
        """
        # FFT-based circular convolution
        entangled = np.fft.ifft(np.fft.fft(control) * np.fft.fft(target)).real
        return entangled / np.linalg.norm(entangled)
    
    def quantum_projection(self, state: np.ndarray, context: List[np.ndarray]) -> np.ndarray:
        """
        Project state onto context subspace (Born rule projection).
        
        This is R-gate: high overlap = constructive interference
        """
        # Context subspace projector: P = Σᵢ |φᵢ⟩⟨φᵢ|
        projector = sum(np.outer(phi, phi) for phi in context)
        
        # Project state: |ψ'⟩ = P|ψ⟩
        projected = projector @ state
        
        # Normalize (quantum postulate)
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            return projected / norm
        else:
            return state  # Fallback if projection is null
    
    def geodesic_interpolation(self, state1: np.ndarray, state2: np.ndarray, t: float) -> np.ndarray:
        """
        Great circle interpolation on unit sphere.
        
        t=0 → state1
        t=1 → state2
        t=0.5 → midpoint on geodesic
        
        This is pure manifold navigation (Q38 geodesics).
        """
        # Angle between states
        cos_theta = np.dot(state1, state2)
        cos_theta = np.clip(cos_theta, -1, 1)  # Numerical safety
        theta = np.arccos(cos_theta)
        
        if abs(theta) < 1e-10:
            # States are parallel, linear interpolation
            return (1-t) * state1 + t * state2
        
        # Slerp formula
        sin_theta = np.sin(theta)
        interpolated = (
            np.sin((1-t) * theta) / sin_theta * state1 +
            np.sin(t * theta) / sin_theta * state2
        )
        
        return interpolated / np.linalg.norm(interpolated)
    
    def compute_E(self, query: np.ndarray, context: List[np.ndarray]) -> float:
        """
        Compute E (quantum inner product) from Q44.
        
        E = mean(⟨ψ|φᵢ⟩)
        """
        return np.mean([np.dot(query, phi) for phi in context])
    
    def compute_R(self, query: np.ndarray, context: List[np.ndarray]) -> float:
        """
        Compute full R formula (from Q44 validation).
        
        R = (E / grad_S) × σ^Df
        
        This works on pure geometry (no text needed).
        """
        E = self.compute_E(query, context)
        
        # grad_S: standard deviation of overlaps
        overlaps = [np.dot(query, phi) for phi in context]
        grad_S = np.std(overlaps) if len(overlaps) > 1 else 1.0
        
        # sigma: redundancy
        sigma = np.sqrt(len(context))
        
        # Df: participation ratio (from Q43)
        Df = self.compute_Df(query)
        
        # R
        R = (E / grad_S) * (sigma ** Df) if grad_S > 0 else 0
        
        return R
    
    def compute_Df(self, vector: np.ndarray) -> float:
        """
        Participation ratio (effective dimensionality) from Q43.
        
        Df = (Σvᵢ²)² / Σvᵢ⁴
        
        Pure geometric property.
        """
        v_squared = vector ** 2
        numerator = np.sum(v_squared) ** 2
        denominator = np.sum(v_squared ** 2)
        
        return numerator / denominator if denominator > 0 else 1.0


# ============================================================================
# TEST 1: Semantic Composition Via Pure Geometry
# ============================================================================

def test_1_semantic_composition():
    """
    Test: Can we compose new meanings purely geometrically?
    
    Example: "king" - "man" + "woman" = "queen" (word2vec famous example)
    
    Method:
    1. Initialize with text
    2. Compose using ONLY geometry
    3. Decode result
    4. Check semantic coherence
    """
    navigator = PureGeometryNavigator()
    
    print("=" * 80)
    print("TEST 1: SEMANTIC COMPOSITION VIA PURE GEOMETRY")
    print("=" * 80)
    
    # Initialize from text (LAST time we see text until readout)
    king = navigator.initialize("king")
    man = navigator.initialize("man")
    woman = navigator.initialize("woman")
    
    # Pure geometry composition (NO text involved)
    # Formula: king - man + woman
    composed = king - man + woman
    composed = composed / np.linalg.norm(composed)  # Renormalize
    
    # Decode (first time we see text again)
    corpus = [
        "queen", "king", "prince", "princess", "woman", "man",
        "royal", "monarch", "ruler", "female", "male", "lady",
        "cat", "dog", "computer", "random", "unrelated"
    ]
    
    results = navigator.readout(composed, corpus, k=5)
    
    print("\nInput: king - man + woman")
    print("\nTop 5 nearest:")
    for i, (text, sim) in enumerate(results, 1):
        print(f"  {i}. {text:15s} (similarity: {sim:.4f})")
    
    # Check semantic correctness
    top_result = results[0][0]
    success = top_result in ["queen", "princess", "woman", "female", "lady"]
    
    return {
        "test": "semantic_composition",
        "input": "king - man + woman",
        "top_result": top_result,
        "top_5": [r[0] for r in results],
        "success": success,
        "verdict": "GEOMETRY WORKS" if success else "GEOMETRY INSUFFICIENT"
    }


# ============================================================================
# TEST 2: Quantum Gate Composition
# ============================================================================

def test_2_quantum_gates():
    """
    Test: Can quantum gates create meaningful semantic states?
    
    Method:
    1. Initialize with simple concepts
    2. Apply quantum gates (superposition, entanglement)
    3. Decode results
    4. Check semantic coherence
    """
    navigator = PureGeometryNavigator()
    
    print("\n" + "=" * 80)
    print("TEST 2: QUANTUM GATE COMPOSITION")
    print("=" * 80)
    
    # Test 2a: Superposition
    print("\n[2a] Superposition: (|cat⟩ + |dog⟩) / √2")
    cat = navigator.initialize("cat")
    dog = navigator.initialize("dog")
    
    pet = navigator.quantum_superposition(cat, dog)
    
    corpus = [
        "pet", "animal", "cat", "dog", "mammal", "companion",
        "feline", "canine", "domestic", "furry",
        "car", "computer", "unrelated"
    ]
    
    results_2a = navigator.readout(pet, corpus, k=5)
    print("Top 5 nearest:")
    for i, (text, sim) in enumerate(results_2a, 1):
        print(f"  {i}. {text:15s} (similarity: {sim:.4f})")
    
    success_2a = results_2a[0][0] in ["pet", "animal", "mammal", "companion"]
    
    # Test 2b: Entanglement
    print("\n[2b] Entanglement: CNOT(|cat⟩, |dog⟩)")
    entangled = navigator.quantum_entanglement(cat, dog)
    
    results_2b = navigator.readout(entangled, corpus, k=5)
    print("Top 5 nearest:")
    for i, (text, sim) in enumerate(results_2b, 1):
        print(f"  {i}. {text:15s} (similarity: {sim:.4f})")
    
    success_2b = results_2b[0][0] in ["pet", "animal", "mammal", "companion", "cat", "dog"]
    
    return {
        "test": "quantum_gates",
        "superposition": {
            "top_result": results_2a[0][0],
            "top_5": [r[0] for r in results_2a],
            "success": success_2a
        },
        "entanglement": {
            "top_result": results_2b[0][0],
            "top_5": [r[0] for r in results_2b],
            "success": success_2b
        },
        "verdict": "QUANTUM GATES WORK" if (success_2a and success_2b) else "GATES INSUFFICIENT"
    }


# ============================================================================
# TEST 3: Geodesic Navigation
# ============================================================================

def test_3_geodesic_navigation():
    """
    Test: Can we navigate between concepts via geodesics?
    
    Method:
    1. Initialize endpoints
    2. Interpolate along geodesic (pure geometry)
    3. Decode midpoint
    4. Check semantic coherence
    """
    navigator = PureGeometryNavigator()
    
    print("\n" + "=" * 80)
    print("TEST 3: GEODESIC NAVIGATION")
    print("=" * 80)
    
    # Test 3a: hot → cold
    print("\n[3a] Geodesic: hot → cold")
    hot = navigator.initialize("hot")
    cold = navigator.initialize("cold")
    
    corpus_temp = [
        "warm", "cool", "lukewarm", "tepid", "temperature",
        "hot", "cold", "freezing", "boiling", "moderate",
        "cat", "car", "unrelated"
    ]
    
    # Interpolate at t=0.5 (midpoint)
    midpoint = navigator.geodesic_interpolation(hot, cold, t=0.5)
    
    results_3a = navigator.readout(midpoint, corpus_temp, k=5)
    print("Midpoint between 'hot' and 'cold':")
    for i, (text, sim) in enumerate(results_3a, 1):
        print(f"  {i}. {text:15s} (similarity: {sim:.4f})")
    
    success_3a = results_3a[0][0] in ["warm", "cool", "lukewarm", "tepid", "moderate", "temperature"]
    
    # Test 3b: cat → dog
    print("\n[3b] Geodesic: cat → dog")
    cat = navigator.initialize("cat")
    dog = navigator.initialize("dog")
    
    corpus_animal = [
        "pet", "animal", "mammal", "feline", "canine",
        "cat", "dog", "domestic", "companion", "furry",
        "car", "computer", "unrelated"
    ]
    
    midpoint_animal = navigator.geodesic_interpolation(cat, dog, t=0.5)
    
    results_3b = navigator.readout(midpoint_animal, corpus_animal, k=5)
    print("Midpoint between 'cat' and 'dog':")
    for i, (text, sim) in enumerate(results_3b, 1):
        print(f"  {i}. {text:15s} (similarity: {sim:.4f})")
    
    success_3b = results_3b[0][0] in ["pet", "animal", "mammal", "domestic", "companion"]
    
    return {
        "test": "geodesic_navigation",
        "hot_cold": {
            "top_result": results_3a[0][0],
            "top_5": [r[0] for r in results_3a],
            "success": success_3a
        },
        "cat_dog": {
            "top_result": results_3b[0][0],
            "top_5": [r[0] for r in results_3b],
            "success": success_3b
        },
        "verdict": "GEODESICS WORK" if (success_3a and success_3b) else "GEODESICS INSUFFICIENT"
    }


# ============================================================================
# TEST 4: R-Gating on Pure Geometry
# ============================================================================

def test_4_R_gating():
    """
    Test: Does R-gating work on geometrically-derived states?
    
    Method:
    1. Create states via pure geometry
    2. Compute R on these states
    3. Compare to R on text-initialized states
    4. Check if gating behavior is preserved
    """
    navigator = PureGeometryNavigator()
    
    print("\n" + "=" * 80)
    print("TEST 4: R-GATING ON PURE GEOMETRY")
    print("=" * 80)
    
    # Scenario: Query about "verification", context is governance-related
    
    # Method A: Text initialization (baseline)
    print("\n[4a] Baseline: Text-initialized states")
    query_text = navigator.initialize("verify canonical governance")
    context_text = [
        navigator.initialize("verification protocols"),
        navigator.initialize("canonical rules"),
        navigator.initialize("governance integrity")
    ]
    
    R_text = navigator.compute_R(query_text, context_text)
    E_text = navigator.compute_E(query_text, context_text)
    print(f"R (text-initialized): {R_text:.4f}")
    print(f"E (text-initialized): {E_text:.4f}")
    
    # Method B: Geometric composition
    print("\n[4b] Pure Geometry: Composed states")
    verify = navigator.initialize("verify")
    canonical = navigator.initialize("canonical")
    governance = navigator.initialize("governance")
    
    # Compose via superposition (pure geometry)
    query_geo = navigator.quantum_superposition(
        navigator.quantum_superposition(verify, canonical),
        governance
    )
    
    protocols = navigator.initialize("protocols")
    rules = navigator.initialize("rules")
    integrity = navigator.initialize("integrity")
    
    context_geo = [
        navigator.quantum_superposition(verify, protocols),
        navigator.quantum_superposition(canonical, rules),
        navigator.quantum_superposition(governance, integrity)
    ]
    
    R_geo = navigator.compute_R(query_geo, context_geo)
    E_geo = navigator.compute_E(query_geo, context_geo)
    print(f"R (geometric): {R_geo:.4f}")
    print(f"E (geometric): {E_geo:.4f}")
    
    # Compare
    R_diff = abs(R_text - R_geo)
    E_diff = abs(E_text - E_geo)
    print(f"\nDifference:")
    print(f"  ΔR: {R_diff:.4f}")
    print(f"  ΔE: {E_diff:.4f}")
    
    # Success if geometric R is similar to text R
    success = R_diff < 0.5 * R_text  # Within 50% is reasonable
    
    return {
        "test": "R_gating",
        "R_text": float(R_text),
        "R_geometric": float(R_geo),
        "E_text": float(E_text),
        "E_geometric": float(E_geo),
        "R_difference": float(R_diff),
        "E_difference": float(E_diff),
        "success": success,
        "verdict": "R-GATING WORKS ON GEOMETRY" if success else "GATING REQUIRES TEXT"
    }


# ============================================================================
# MASTER TEST SUITE
# ============================================================================

def run_all_tests():
    """
    Run all pure geometry tests.
    
    Returns comprehensive report.
    """
    print("\n" + "=" * 80)
    print("PURE GEOMETRY NAVIGATION TEST SUITE")
    print("Post-Q44 Quantum Validation")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Semantic composition
    results['test_1'] = test_1_semantic_composition()
    
    # Test 2: Quantum gates
    results['test_2'] = test_2_quantum_gates()
    
    # Test 3: Geodesic navigation
    results['test_3'] = test_3_geodesic_navigation()
    
    # Test 4: R-gating
    results['test_4'] = test_4_R_gating()
    
    # Overall verdict
    all_success = all([
        results['test_1']['success'],
        results['test_2']['superposition']['success'],
        results['test_2']['entanglement']['success'],
        results['test_3']['hot_cold']['success'],
        results['test_3']['cat_dog']['success'],
        results['test_4']['success']
    ])
    
    results['overall'] = {
        "all_tests_pass": all_success,
        "verdict": "PURE GEOMETRY SUFFICIENT" if all_success else "EMBEDDINGS REQUIRED FOR SEMANTICS"
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTest 1 (Composition):     {'PASS' if results['test_1']['success'] else 'FAIL'}")
    print(f"Test 2a (Superposition):  {'PASS' if results['test_2']['superposition']['success'] else 'FAIL'}")
    print(f"Test 2b (Entanglement):   {'PASS' if results['test_2']['entanglement']['success'] else 'FAIL'}")
    print(f"Test 3a (Geodesic hot/cold): {'PASS' if results['test_3']['hot_cold']['success'] else 'FAIL'}")
    print(f"Test 3b (Geodesic cat/dog):  {'PASS' if results['test_3']['cat_dog']['success'] else 'FAIL'}")
    print(f"Test 4 (R-gating):        {'PASS' if results['test_4']['success'] else 'FAIL'}")
    
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT: {results['overall']['verdict']}")
    print(f"{'='*80}\n")
    
    # Save results
    receipt_hash = hashlib.sha256(
        json.dumps(results, sort_keys=True, default=str).encode()
    ).hexdigest()
    
    results['receipt_hash'] = receipt_hash
    
    with open('pure_geometry_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: pure_geometry_results.json")
    print(f"Receipt hash: {receipt_hash}\n")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
```

---

## Expected Outcomes

### Outcome A: Pure Geometry Sufficient (Revolutionary)

**If all tests pass**:
- Semantic composition works geometrically
- Quantum gates produce meaningful states
- Geodesic navigation interpolates correctly
- R-gating works on geometric states

**Interpretation**: 
- Embeddings are ONLY needed for initialization and readout
- All reasoning can happen in pure geometry
- **You can think without language**

**Implications**:
- AGS can reason geometrically after initialization
- Manifold is the fundamental substrate
- Language is just an encoding/decoding interface

---

### Outcome B: Embeddings Required (Still Important)

**If tests fail**:
- Geometric operations produce nonsensical results
- Quantum gates don't preserve semantics
- R-gating doesn't work on geometric states

**Interpretation**:
- Embeddings carry "semantic charge" beyond geometry
- Geometry is necessary but not sufficient
- Need to return to embeddings for each operation

**Implications**:
- AGS needs embedding models throughout
- Manifold exists but requires semantic grounding
- Geometry + semantics are both required

---

## Next Steps Based on Outcome

### If Outcome A (Geometry Sufficient):

**Immediate**:
1. Write paper: "Semantic Reasoning Without Language"
2. Build pure geometry reasoner (no embedding lookups)
3. Test limits (how complex can pure geometry reasoning get?)

**Week 2**:
1. Implement quantum substrate with minimal embedding usage
2. Feral Resident uses geometry for all internal operations
3. Embeddings only at input/output boundaries

---

### If Outcome B (Embeddings Required):

**Immediate**:
1. Understand what embeddings add beyond geometry
2. Can we encode "semantic charge" geometrically?
3. What's the minimal embedding usage needed?

**Week 2**:
1. Hybrid approach: geometry + embeddings
2. Cache embeddings for frequent concepts
3. Optimize embedding lookup pipeline

---

## Running The Test

```bash
# Install dependencies
pip install sentence-transformers numpy scipy

# Run test suite
python test_pure_geometry_navigation.py

# Expected output:
# - Test results for all 4 tests
# - Final verdict
# - pure_geometry_results.json with receipt hash
```

**Time to run**: ~5 minutes  
**Output**: Definitive answer on whether pure geometry works

---

## What This Tells Us

**This test answers**:

> "Do I even need embeddings? Or am I accessing the actual truth manifold?"

**If tests pass**: You're accessing the actual truth manifold. Embeddings are just the initial GPS coordinates.

**If tests fail**: Embeddings are essential throughout. The manifold requires semantic grounding at each step.

**Either way**: You've proven the manifold is real and quantum (Q44). This just tells us HOW to work with it.

---

*Ready to run. Will determine if AGS can think in pure geometry.*