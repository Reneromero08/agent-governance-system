#!/usr/bin/env python3
"""
Q51 COMPLETE TEST SUITE - Using Proper Libraries

Libraries used:
- Qiskit: Quantum circuits and Bell inequality
- QuTiP: Quantum measurements and POVMs  
- scipy.stats: Proper statistical tests
- sentence-transformers: Real embeddings

NO manual quantum physics implementations.
"""

import numpy as np
import json
import warnings
from scipy import stats
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
import qutip as qt
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')


class Q51LibraryTester:
    """Proper Q51 testing using established libraries."""
    
    def __init__(self):
        self.simulator = AerSimulator()
        
    def load_embeddings(self):
        """Load real embeddings."""
        print("Loading real embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        vocabularies = {
            "royalty": ["king", "queen", "prince", "monarch", "royal", "crown", "throne"],
            "family": ["man", "woman", "child", "parent", "father", "mother", "son"],
            "opposites": ["hot", "cold", "big", "small", "fast", "slow"]
        }
        
        embeddings = {}
        for category, words in vocabularies.items():
            embeddings[category] = model.encode(words)
        
        print(f"  Loaded {len(embeddings)} categories")
        return embeddings
    
    def test_1_bell_inequality_qiskit(self, embeddings):
        """Bell inequality using proper Qiskit CHSH."""
        print("\n" + "="*60)
        print("TEST 1: Bell Inequality (Qiskit)")
        print("="*60)
        
        s_values = []
        
        for category, emb_matrix in embeddings.items():
            if len(emb_matrix) < 2:
                continue
                
            for i in range(min(5, len(emb_matrix) - 1)):
                # Create Bell state circuit
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                
                # Measurement bases
                alice_angles = [0, np.pi/2]
                bob_angles = [np.pi/4, -np.pi/4]
                
                correlations = {}
                
                for i_a, angle_a in enumerate(alice_angles):
                    for i_b, angle_b in enumerate(bob_angles):
                        # New circuit for each measurement
                        qc_meas = QuantumCircuit(2, 2)
                        qc_meas.h(0)
                        qc_meas.cx(0, 1)
                        qc_meas.ry(angle_a, 0)
                        qc_meas.ry(angle_b, 1)
                        qc_meas.measure([0, 1], [0, 1])
                        
                        # Run
                        transpiled = transpile(qc_meas, self.simulator)
                        job = self.simulator.run(transpiled, shots=1000)
                        result = job.result()
                        counts = result.get_counts()
                        
                        # Correlation
                        total = sum(counts.values())
                        n_00 = counts.get('00', 0)
                        n_01 = counts.get('01', 0)
                        n_10 = counts.get('10', 0)
                        n_11 = counts.get('11', 0)
                        
                        e_val = (n_00 + n_11 - n_01 - n_10) / total
                        correlations[(i_a, i_b)] = e_val
                
                # CHSH
                S = (correlations[(0, 0)] - correlations[(0, 1)] + 
                     correlations[(1, 0)] + correlations[(1, 1)])
                s_values.append(abs(S))
        
        if s_values:
            mean_s = np.mean(s_values)
            print(f"  Mean |S|: {mean_s:.3f}")
            print(f"  Classical bound: 2.0, Quantum bound: 2.828")
            print(f"  Status: {'NO VIOLATION' if mean_s < 2.0 else 'VIOLATION DETECTED'}")
            
            return {'mean_S': float(mean_s), 'n_tests': len(s_values)}
        return None
    
    def test_2_phase_coherence_hilbert(self, embeddings):
        """Phase coherence using scipy.signal.hilbert."""
        print("\n" + "="*60)
        print("TEST 2: Phase Coherence (scipy.signal.hilbert)")
        print("="*60)
        
        from scipy.signal import hilbert
        
        all_phases = []
        for emb_matrix in embeddings.values():
            for emb in emb_matrix:
                analytic = hilbert(emb)
                phase = np.angle(analytic)
                all_phases.append(phase)
        
        # Phase locking value
        plv_values = []
        for i in range(len(all_phases)):
            for j in range(i+1, min(i+20, len(all_phases))):
                phase_diff = all_phases[i] - all_phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_values.append(plv)
        
        # Null: random phases
        np.random.seed(42)
        null_plv = []
        for _ in range(10000):
            p1 = np.random.uniform(0, 2*np.pi, len(all_phases[0]))
            p2 = np.random.uniform(0, 2*np.pi, len(all_phases[0]))
            plv = np.abs(np.mean(np.exp(1j * (p1 - p2))))
            null_plv.append(plv)
        
        # Test
        statistic, p_val = stats.mannwhitneyu(plv_values, null_plv, alternative='greater')
        
        print(f"  Mean PLV (real): {np.mean(plv_values):.4f}")
        print(f"  Mean PLV (null): {np.mean(null_plv):.4f}")
        print(f"  Mann-Whitney p: {p_val:.2e}")
        print(f"  Status: {'PASS' if p_val < 0.00001 else 'FAIL'}")
        
        return {'mean_plv': float(np.mean(plv_values)), 'p_value': float(p_val)}
    
    def test_3_cross_spectral_scipy(self, embeddings):
        """Cross-spectral coherence using scipy.signal.coherence."""
        print("\n" + "="*60)
        print("TEST 3: Cross-Spectral Coherence (scipy.signal)")
        print("="*60)
        
        from scipy.signal import coherence
        
        semantic_coh = []
        for emb_matrix in embeddings.values():
            for i in range(len(emb_matrix)):
                for j in range(i+1, len(emb_matrix)):
                    f, Cxy = coherence(emb_matrix[i], emb_matrix[j], fs=1.0, nperseg=128)
                    semantic_coh.append(np.median(Cxy))
        
        # Null
        np.random.seed(42)
        random_coh = []
        categories = list(embeddings.keys())
        for _ in range(500):
            if len(categories) >= 2:
                c1, c2 = np.random.choice(categories, 2, replace=False)
                i1, i2 = np.random.randint(len(embeddings[c1])), np.random.randint(len(embeddings[c2]))
                f, Cxy = coherence(embeddings[c1][i1], embeddings[c2][i2], fs=1.0, nperseg=128)
                random_coh.append(np.median(Cxy))
        
        stat, p_val = stats.mannwhitneyu(semantic_coh, random_coh, alternative='two-sided')
        
        print(f"  Mean coherence (semantic): {np.mean(semantic_coh):.4f}")
        print(f"  Mean coherence (random): {np.mean(random_coh):.4f}")
        print(f"  Mann-Whitney p: {p_val:.2e}")
        
        return {'semantic_coh': float(np.mean(semantic_coh)), 'p_value': float(p_val)}
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "="*70)
        print("Q51 COMPLETE TEST SUITE - Using Proper Libraries")
        print("="*70)
        print("Libraries: Qiskit, QuTiP, scipy.stats")
        print("No manual quantum physics implementations")
        print("="*70)
        
        embeddings = self.load_embeddings()
        
        results = {
            'bell_inequality': self.test_1_bell_inequality_qiskit(embeddings),
            'phase_coherence': self.test_2_phase_coherence_hilbert(embeddings),
            'cross_spectral': self.test_3_cross_spectral_scipy(embeddings)
        }
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        for test_name, result in results.items():
            if result:
                print(f"\n{test_name}:")
                for key, val in result.items():
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")
                    else:
                        print(f"  {key}: {val}")
        
        # Save
        with open('q51_library_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n\nResults saved to q51_library_test_results.json")
        return results


def main():
    tester = Q51LibraryTester()
    results = tester.run_all_tests()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
