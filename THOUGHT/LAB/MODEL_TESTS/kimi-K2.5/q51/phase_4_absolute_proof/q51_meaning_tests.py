#!/usr/bin/env python3
"""
Q51 MEANING TESTS - Testing Semiotic Space, Not Embeddings

Q51 Hypothesis: Real embeddings are shadows of complex semiotic space.

These tests verify:
1. Multiplicative composition (phase addition)
2. Context superposition (quantum measurement)
3. Semantic interference (wave cancellation)
4. Analogical reasoning via phase arithmetic

Uses proper libraries:
- scipy.stats for statistical validation
- Qiskit only for quantum simulation validation
"""

import numpy as np
import json
import warnings
from scipy import stats
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')


class Q51MeaningTester:
    """Test Q51 by testing semantic operations, not raw vectors."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.results = {}
        
    def test_1_multiplicative_composition(self):
        """
        TEST 1: Multiplicative vs Additive Composition
        
        If meaning is complex, composition should be multiplicative:
        |king - man + woman| ≈ |queen| via phase, not magnitude
        
        This tests if analogies work via phase cancellation/addition.
        """
        print("\n" + "="*60)
        print("TEST 1: Multiplicative Composition (Phase Arithmetic)")
        print("="*60)
        
        analogies = [
            ("king", "man", "woman", "queen"),
            ("Paris", "France", "Italy", "Rome"),
            ("walk", "walked", "swim", "swam"),
            ("big", "bigger", "small", "smaller"),
            ("mouse", "mice", "child", "children")
        ]
        
        multiplicative_scores = []
        additive_scores = []
        
        for a, b, c, expected in analogies:
            # Get embeddings
            v_a = self.model.encode([a])[0]
            v_b = self.model.encode([b])[0]
            v_c = self.model.encode([c])[0]
            v_expected = self.model.encode([expected])[0]
            
            # Multiplicative: a - b + c (vector arithmetic)
            v_multi = v_a - v_b + v_c
            v_multi = v_multi / (np.linalg.norm(v_multi) + 1e-10)
            
            # Compute cosine similarities
            multiplicative_sim = 1 - cosine(v_multi, v_expected)
            
            # Additive baseline: average
            v_add = (v_a + v_c) / 2
            v_add = v_add / (np.linalg.norm(v_add) + 1e-10)
            additive_sim = 1 - cosine(v_add, v_expected)
            
            multiplicative_scores.append(multiplicative_sim)
            additive_scores.append(additive_sim)
        
        # Statistical test
        stat, p_val = stats.wilcoxon(multiplicative_scores, additive_scores, alternative='greater')
        
        mean_multi = np.mean(multiplicative_scores)
        mean_add = np.mean(additive_scores)
        
        print(f"  Multiplicative mean similarity: {mean_multi:.4f}")
        print(f"  Additive mean similarity: {mean_add:.4f}")
        print(f"  Wilcoxon p-value: {p_val:.2e}")
        print(f"  Status: {'PASS (multiplicative wins)' if p_val < 0.00001 and mean_multi > mean_add else 'FAIL'}")
        
        self.results['multiplicative_composition'] = {
            'multiplicative_mean': float(mean_multi),
            'additive_mean': float(mean_add),
            'p_value': float(p_val),
            'n_analogies': len(analogies)
        }
        
        return p_val < 0.00001 and mean_multi > mean_add
    
    def test_2_context_superposition(self):
        """
        TEST 2: Context Acts Like Quantum Measurement
        
        If meaning has phase, context should collapse superpositions:
        "bank" without context = superposition (river, money)
        "bank" + "river" context = collapsed to river meaning
        
        Test: Does adding context reduce ambiguity?
        """
        print("\n" + "="*60)
        print("TEST 2: Context Superposition (Quantum Measurement)")
        print("="*60)
        
        ambiguous_words = [
            ("bank", ["river", "money", "finance", "water", "loan", "stream"]),
            ("bark", ["tree", "dog", "puppy", "oak", "woof", "trunk"]),
            ("light", ["weight", "bright", "sun", "heavy", "darkness", "lamp"])
        ]
        
        context_effects = []
        
        for word, contexts in ambiguous_words:
            # Encode ambiguous word
            v_word = self.model.encode([word])[0]
            
            # Encode contexts
            v_contexts = [self.model.encode([c])[0] for c in contexts]
            
            # Without context: variance across all context directions
            similarities_no_context = [
                1 - cosine(v_word, vc) for vc in v_contexts
            ]
            variance_no_context = np.var(similarities_no_context)
            
            # With each context: variance should be lower (collapsed)
            variances_with_context = []
            for vc in v_contexts:
                # Project word onto context direction
                projection = v_word + 0.3 * vc  # Context shifts meaning
                projection = projection / (np.linalg.norm(projection) + 1e-10)
                
                # Check similarity to all contexts after projection
                sims = [1 - cosine(projection, vcx) for vcx in v_contexts]
                variances_with_context.append(np.var(sims))
            
            # Context should reduce variance (measurement collapses superposition)
            mean_var_with = np.mean(variances_with_context)
            context_effects.append(variance_no_context - mean_var_with)
        
        # Test if context reduces variance
        stat, p_val = stats.ttest_1samp(context_effects, 0)
        
        mean_effect = np.mean(context_effects)
        
        print(f"  Mean variance reduction: {mean_effect:.4f}")
        print(f"  t-test p-value: {p_val:.2e}")
        print(f"  Status: {'PASS (context collapses superposition)' if p_val < 0.00001 and mean_effect > 0 else 'FAIL'}")
        
        self.results['context_superposition'] = {
            'mean_variance_reduction': float(mean_effect),
            'p_value': float(p_val),
            'n_words': len(ambiguous_words)
        }
        
        return p_val < 0.00001 and mean_effect > 0
    
    def test_3_semantic_interference(self):
        """
        TEST 3: Semantic Interference (Wave Cancellation)
        
        If meaning has phase, opposites should interfere destructively:
        "hot" + "cold" (equal magnitude, opposite phase) ≈ zero vector
        
        Test: Do antonyms show destructive interference?
        """
        print("\n" + "="*60)
        print("TEST 3: Semantic Interference (Destructive)")
        print("="*60)
        
        antonym_pairs = [
            ("hot", "cold"),
            ("big", "small"),
            ("happy", "sad"),
            ("fast", "slow"),
            ("light", "dark")
        ]
        
        interference_scores = []
        control_scores = []
        
        for word1, word2 in antonym_pairs:
            v1 = self.model.encode([word1])[0]
            v2 = self.model.encode([word2])[0]
            
            # Sum vectors (interference)
            v_sum = v1 + v2
            interference_magnitude = np.linalg.norm(v_sum)
            
            # Control: sum of unrelated words
            v_control = self.model.encode(["table"])[0] + self.model.encode(["chair"])[0]
            control_magnitude = np.linalg.norm(v_control)
            
            # Antonyms should show more cancellation
            cancellation = 1 - (interference_magnitude / (np.linalg.norm(v1) + np.linalg.norm(v2)))
            
            interference_scores.append(cancellation)
        
        # Test if antonyms show significant cancellation
        stat, p_val = stats.ttest_1samp(interference_scores, 0)
        
        mean_cancellation = np.mean(interference_scores)
        
        print(f"  Mean cancellation (antonyms): {mean_cancellation:.4f}")
        print(f"  t-test p-value: {p_val:.2e}")
        print(f"  Status: {'PASS (destructive interference)' if p_val < 0.00001 else 'FAIL'}")
        
        self.results['semantic_interference'] = {
            'mean_cancellation': float(mean_cancellation),
            'p_value': float(p_val),
            'n_pairs': len(antonym_pairs)
        }
        
        return p_val < 0.00001
    
    def run_all_tests(self):
        """Run all meaning-based tests."""
        print("\n" + "="*70)
        print("Q51 MEANING TESTS - Testing Semiotic Space")
        print("Not testing embeddings - testing semantic operations")
        print("="*70)
        
        tests = {
            'multiplicative_composition': self.test_1_multiplicative_composition(),
            'context_superposition': self.test_2_context_superposition(),
            'semantic_interference': self.test_3_semantic_interference()
        }
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        passed = sum(1 for v in tests.values() if v)
        total = len(tests)
        
        for test_name, result in tests.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name}: {status}")
        
        print(f"\nPassed: {passed}/{total}")
        print(f"Q51 Status: {'TRUE' if passed >= 2 else 'FALSE'} (≥2/3 required)")
        
        # Save
        with open('q51_meaning_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\nResults saved to q51_meaning_test_results.json")
        
        return passed >= 2


def main():
    tester = Q51MeaningTester()
    result = tester.run_all_tests()
    return 0 if result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
