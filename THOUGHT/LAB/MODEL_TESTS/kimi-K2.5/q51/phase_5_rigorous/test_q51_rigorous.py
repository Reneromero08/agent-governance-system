#!/usr/bin/env python3
"""
Q51 PHASE 5: Rigorous Complex Plane Test

**Hypothesis:** Real embeddings are shadows of complex semiotic space

**Test:** If true, we can extract phase angles via PCA to 2D and test phase arithmetic:
- Analogies should satisfy: θ_a - θ_b ≈ θ_c - θ_d (phase differences preserved)
- Antonyms should differ by π (180°) in phase
- Context should rotate phase predictably

**Methodology:**
1. Use ONLY real embeddings from sentence-transformers
2. Project to 2D complex plane via PCA
3. Extract phase angles (θ = atan2(imag, real))
4. Test phase relationships statistically
5. Proper null models and effect sizes

**No synthetic data. No time-series tools. Real physics only.**
"""

import numpy as np
import json
from scipy import stats
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class Q51RigorousTester:
    """Rigorous test of Q51 using real embeddings and proper phase analysis."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.results = {}
        
    def project_to_complex_plane(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings to 2D complex plane via PCA.
        Returns complex numbers: z = x + iy
        """
        # Center the data
        centered = embeddings - np.mean(embeddings, axis=0)
        
        # PCA to 2 dimensions
        pca = PCA(n_components=2)
        projected = pca.fit_transform(centered)
        
        # Convert to complex: z = x + iy
        complex_vals = projected[:, 0] + 1j * projected[:, 1]
        
        return complex_vals
    
    def extract_phases(self, words: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings and extract phases.
        Returns: (complex_values, phase_angles)
        """
        # Get real embeddings
        embeddings = self.model.encode(words)
        
        # Project to complex plane
        complex_vals = self.project_to_complex_plane(embeddings)
        
        # Extract phases
        phases = np.angle(complex_vals)
        
        return complex_vals, phases
    
    def test_1_phase_arithmetic(self) -> Dict:
        """
        TEST 1: Phase Arithmetic for Analogies
        
        If Q51 is true: θ_king - θ_man ≈ θ_queen - θ_woman
        (Phase differences should be preserved across semantic relationships)
        """
        print("\n" + "="*60)
        print("TEST 1: Phase Arithmetic (Analogies)")
        print("="*60)
        
        analogies = [
            ("king", "man", "queen", "woman"),
            ("Paris", "France", "Rome", "Italy"),
            ("walk", "walked", "swim", "swam"),
            ("big", "bigger", "small", "smaller"),
            ("mouse", "mice", "child", "children"),
            ("king", "queen", "man", "woman"),
            ("doctor", "nurse", "man", "woman"),
            ("fast", "faster", "slow", "slower")
        ]
        
        phase_differences = []
        errors = []
        
        for a, b, c, d in analogies:
            # Get phases
            words = [a, b, c, d]
            _, phases = self.extract_phases(words)
            
            # Phase arithmetic: should be (θ_a - θ_b) ≈ (θ_c - θ_d)
            diff_ab = phases[0] - phases[1]
            diff_cd = phases[2] - phases[3]
            
            # Circular difference (handle 2π wraparound)
            error = np.abs(np.angle(np.exp(1j * (diff_ab - diff_cd))))
            
            phase_differences.append((diff_ab, diff_cd))
            errors.append(error)
        
        errors = np.array(errors)
        
        # Statistical test: Are errors smaller than random?
        # Null: Random word pairs should have larger phase errors
        np.random.seed(42)
        random_errors = []
        for _ in range(100):
            # Random words
            random_words = ["table", "chair", "car", "book", "house", "tree", "dog", "cat"]
            sample = np.random.choice(random_words, 4, replace=False)
            _, phases = self.extract_phases(list(sample))
            diff_ab = phases[0] - phases[1]
            diff_cd = phases[2] - phases[3]
            error = np.abs(np.angle(np.exp(1j * (diff_ab - diff_cd))))
            random_errors.append(error)
        
        # Mann-Whitney U test: Are analogy errors smaller than random?
        statistic, p_value = stats.mannwhitneyu(errors, random_errors, alternative='less')
        
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        mean_random = np.mean(random_errors)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(errors) + np.var(random_errors)) / 2)
        cohen_d = (mean_random - mean_error) / pooled_std if pooled_std > 0 else 0
        
        print(f"  Analogy phase errors: mean={mean_error:.3f}, median={median_error:.3f}")
        print(f"  Random phase errors: mean={mean_random:.3f}")
        print(f"  Mann-Whitney p: {p_value:.2e}")
        print(f"  Effect size: d={cohen_d:.3f}")
        
        # Threshold: errors should be < π/4 (45°) for Q51 to hold
        small_errors = np.sum(errors < np.pi/4) / len(errors)
        
        print(f"  Small errors (<45°): {small_errors:.1%}")
        print(f"  Status: {'PASS' if p_value < 0.00001 and cohen_d > 0.5 else 'FAIL'}")
        
        self.results['phase_arithmetic'] = {
            'mean_error': float(mean_error),
            'median_error': float(median_error),
            'random_mean_error': float(mean_random),
            'p_value': float(p_value),
            'cohen_d': float(cohen_d),
            'small_error_rate': float(small_errors),
            'n_analogies': len(analogies)
        }
        
        return self.results['phase_arithmetic']
    
    def test_2_antonym_phase_opposition(self) -> Dict:
        """
        TEST 2: Antonyms Should Differ by π in Phase
        
        If Q51 is true: θ_hot - θ_cold ≈ π (180° phase difference)
        """
        print("\n" + "="*60)
        print("TEST 2: Antonym Phase Opposition")
        print("="*60)
        
        antonyms = [
            ("hot", "cold"),
            ("big", "small"),
            ("happy", "sad"),
            ("fast", "slow"),
            ("light", "dark"),
            ("tall", "short"),
            ("rich", "poor"),
            ("love", "hate")
        ]
        
        phase_diffs = []
        
        for word1, word2 in antonyms:
            words = [word1, word2]
            _, phases = self.extract_phases(words)
            diff = np.abs(phases[0] - phases[1])
            # Circular distance
            diff = min(diff, 2*np.pi - diff)
            phase_diffs.append(diff)
        
        phase_diffs = np.array(phase_diffs)
        
        # Test if antonyms differ by ~π (180°)
        distances_from_pi = np.abs(phase_diffs - np.pi)
        
        # Compare to random word pairs
        np.random.seed(42)
        random_diffs = []
        random_words = ["table", "chair", "car", "book", "house", "tree", "dog", "cat", 
                       "computer", "phone", "door", "window", "pen", "paper"]
        for _ in range(100):
            pair = np.random.choice(random_words, 2, replace=False)
            _, phases = self.extract_phases(list(pair))
            diff = np.abs(phases[0] - phases[1])
            diff = min(diff, 2*np.pi - diff)
            random_diffs.append(diff)
        
        # Are antonyms closer to π than random pairs?
        statistic, p_value = stats.mannwhitneyu(
            distances_from_pi, 
            np.abs(np.array(random_diffs) - np.pi),
            alternative='less'
        )
        
        mean_diff = np.mean(phase_diffs)
        mean_distance_from_pi = np.mean(distances_from_pi)
        
        # Effect size
        pooled_std = np.sqrt((np.var(distances_from_pi) + np.var(np.abs(np.array(random_diffs) - np.pi))) / 2)
        cohen_d = (np.mean(np.abs(np.array(random_diffs) - np.pi)) - mean_distance_from_pi) / pooled_std if pooled_std > 0 else 0
        
        print(f"  Mean phase difference: {mean_diff:.3f} (target: π={np.pi:.3f})")
        print(f"  Mean distance from π: {mean_distance_from_pi:.3f}")
        print(f"  Mann-Whitney p: {p_value:.2e}")
        print(f"  Effect size: d={cohen_d:.3f}")
        
        # Check if any are close to π (within π/4)
        close_to_pi = np.sum(distances_from_pi < np.pi/4) / len(distances_from_pi)
        
        print(f"  Close to π (<45°): {close_to_pi:.1%}")
        print(f"  Status: {'PASS' if p_value < 0.00001 and cohen_d > 0.5 else 'FAIL'}")
        
        self.results['antonym_opposition'] = {
            'mean_phase_diff': float(mean_diff),
            'mean_distance_from_pi': float(mean_distance_from_pi),
            'p_value': float(p_value),
            'cohen_d': float(cohen_d),
            'close_to_pi_rate': float(close_to_pi),
            'n_antonyms': len(antonyms)
        }
        
        return self.results['antonym_opposition']
    
    def test_3_context_phase_rotation(self) -> Dict:
        """
        TEST 3: Context Rotates Phase Predictably
        
        If Q51 is true: Adding context word should rotate phase by predictable amount
        """
        print("\n" + "="*60)
        print("TEST 3: Context Phase Rotation")
        print("="*60)
        
        # Ambiguous words with contexts
        test_cases = [
            ("bank", ["river", "money", "finance", "water"]),
            ("bark", ["tree", "dog", "puppy", "wood"]),
            ("light", ["weight", "bright", "sun", "lamp"])
        ]
        
        rotation_consistency = []
        
        for word, contexts in test_cases:
            # Get base phase
            _, base_phases = self.extract_phases([word])
            base_phase = base_phases[0]
            
            # Get phases with each context
            context_phases = []
            for ctx in contexts:
                # Get embedding of word in context
                phrase = f"{word} {ctx}"
                _, phases = self.extract_phases([phrase])
                context_phases.append(phases[0])
            
            # Calculate rotations from base
            rotations = [np.angle(np.exp(1j * (cp - base_phase))) for cp in context_phases]
            
            # Test: Are rotations consistent across similar contexts?
            if len(rotations) >= 2:
                rotation_std = np.std(rotations)
                rotation_consistency.append(rotation_std)
        
        # Lower std = more consistent rotation
        mean_consistency = np.mean(rotation_consistency)
        
        # Compare to random phrases
        np.random.seed(42)
        random_consistency = []
        random_words = ["table", "chair", "car", "book", "house", "tree", "dog", "cat"]
        for _ in range(50):
            word = np.random.choice(random_words)
            ctxs = np.random.choice(random_words, 4, replace=False)
            _, base_phases = self.extract_phases([word])
            base_phase = base_phases[0]
            context_phases = []
            for ctx in ctxs:
                phrase = f"{word} {ctx}"
                _, phases = self.extract_phases([phrase])
                context_phases.append(phases[0])
            rotations = [np.angle(np.exp(1j * (cp - base_phase))) for cp in context_phases]
            if len(rotations) >= 2:
                random_consistency.append(np.std(rotations))
        
        # Test: Are semantic contexts more consistent than random?
        statistic, p_value = stats.mannwhitneyu(
            rotation_consistency,
            random_consistency,
            alternative='less'
        )
        
        # Effect size
        pooled_std = np.sqrt((np.var(rotation_consistency) + np.var(random_consistency)) / 2)
        cohen_d = (np.mean(random_consistency) - mean_consistency) / pooled_std if pooled_std > 0 else 0
        
        print(f"  Semantic context rotation std: {mean_consistency:.3f}")
        print(f"  Random context rotation std: {np.mean(random_consistency):.3f}")
        print(f"  Mann-Whitney p: {p_value:.2e}")
        print(f"  Effect size: d={cohen_d:.3f}")
        print(f"  Status: {'PASS' if p_value < 0.00001 and cohen_d > 0.5 else 'FAIL'}")
        
        self.results['context_rotation'] = {
            'semantic_rotation_std': float(mean_consistency),
            'random_rotation_std': float(np.mean(random_consistency)),
            'p_value': float(p_value),
            'cohen_d': float(cohen_d),
            'n_words': len(test_cases)
        }
        
        return self.results['context_rotation']
    
    def run_all_tests(self):
        """Run complete rigorous test suite."""
        print("\n" + "="*70)
        print("Q51 PHASE 5: Rigorous Complex Plane Test")
        print("="*70)
        print("Methodology:")
        print("  - Real embeddings only (sentence-transformers)")
        print("  - PCA projection to 2D complex plane")
        print("  - Phase angle extraction and arithmetic")
        print("  - Proper statistical controls and effect sizes")
        print("  - No synthetic data. No time-series tools.")
        print("="*70)
        
        # Run tests
        test1 = self.test_1_phase_arithmetic()
        test2 = self.test_2_antonym_phase_opposition()
        test3 = self.test_3_context_phase_rotation()
        
        # Summary
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        passed = 0
        for test_name, result in self.results.items():
            p_val = result['p_value']
            cohen_d = result['cohen_d']
            passed_test = p_val < 0.00001 and cohen_d > 0.5
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"  {test_name:25s} {status} (p={p_val:.2e}, d={cohen_d:.2f})")
            if passed_test:
                passed += 1
        
        print(f"\nPassed: {passed}/3 tests")
        
        if passed >= 2:
            verdict = "Q51 = TRUE (Strong evidence for complex phase structure)"
            confidence = "HIGH"
        elif passed == 1:
            verdict = "Q51 = POSSIBLE (Some evidence, needs replication)"
            confidence = "MODERATE"
        else:
            verdict = "Q51 = FALSE (No evidence for complex phase structure)"
            confidence = "HIGH"
        
        print(f"\nVerdict: {verdict}")
        print(f"Confidence: {confidence}")
        print("="*70)
        
        # Save results
        with open('phase_5_rigorous_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\nResults saved to phase_5_rigorous_results.json")
        
        return passed >= 2


def main():
    tester = Q51RigorousTester()
    result = tester.run_all_tests()
    return 0 if result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
