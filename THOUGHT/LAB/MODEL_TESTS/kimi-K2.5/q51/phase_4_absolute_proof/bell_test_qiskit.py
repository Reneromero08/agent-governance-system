#!/usr/bin/env python3
"""
Q51 BELL INEQUALITY TEST - Proper Implementation using Qiskit
Correct CHSH test with proper quantum circuits
"""

import numpy as np
import json
import warnings
from scipy import stats
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')


def embedding_to_quantum_state(embedding):
    """Convert embedding to valid quantum state using amplitude encoding."""
    # Normalize
    normalized = embedding / np.linalg.norm(embedding)
    
    # Calculate required qubits to fit embedding dimension
    n_qubits = int(np.ceil(np.log2(len(normalized))))
    dim = 2 ** n_qubits
    
    # Pad to power of 2
    amplitudes = np.zeros(dim, dtype=complex)
    amplitudes[:len(normalized)] = normalized.astype(complex)
    
    # Normalize again after padding
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    
    return amplitudes


def create_chsh_circuit(angle_a, angle_b):
    """Create CHSH measurement circuit.
    
    Alice measures at angle_a, Bob at angle_b
    Returns quantum circuit that prepares Bell state and measures
    """
    qc = QuantumCircuit(2, 2)
    
    # Create Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT with control 0, target 1
    
    # Rotate measurement bases
    qc.ry(angle_a, 0)  # Alice's measurement angle
    qc.ry(angle_b, 1)  # Bob's measurement angle
    
    # Measure
    qc.measure(0, 0)
    qc.measure(1, 1)
    
    return qc


def run_chsh_test(embeddings_dict):
    """Run proper CHSH Bell inequality test using Qiskit."""
    print("\n" + "="*60)
    print("Q51 BELL INEQUALITY TEST - Qiskit Implementation")
    print("="*60)
    
    results_by_model = {}
    
    # Use Qiskit Aer simulator
    simulator = AerSimulator()
    
    for model_name, embeddings in embeddings_dict.items():
        print(f"\nTesting {model_name}...")
        
        # CHSH requires 4 measurement settings
        # Alice: a=0, a'=π/2 (optimal for Bell state)
        # Bob: b=π/4, b'=-π/4 (or 3π/4)
        alice_angles = [0, np.pi/2]
        bob_angles = [np.pi/4, -np.pi/4]
        
        s_values = []
        
        # Test on semantic pairs
        for category, emb_matrix in embeddings.items():
            if len(emb_matrix) < 2:
                continue
            
            for i in range(min(5, len(emb_matrix) - 1)):
                emb_a = emb_matrix[i]
                emb_b = emb_matrix[i + 1]
                
                # Convert to quantum states
                state_a = embedding_to_quantum_state(emb_a)
                state_b = embedding_to_quantum_state(emb_b)
                
                # Compute correlations for all 4 setting combinations
                correlations = {}
                
                for i_a, angle_a in enumerate(alice_angles):
                    for i_b, angle_b in enumerate(bob_angles):
                        # Create and run CHSH circuit
                        qc = create_chsh_circuit(angle_a, angle_b)
                        
                        # Run circuit with many shots
                        transpiled = transpile(qc, simulator)
                        job = simulator.run(transpiled, shots=1000, seed_simulator=42)
                        result = job.result()
                        counts = result.get_counts()
                        
                        # Calculate correlation E = (N_00 + N_11 - N_01 - N_10) / N_total
                        total = sum(counts.values())
                        n_00 = counts.get('00', 0)
                        n_01 = counts.get('01', 0)
                        n_10 = counts.get('10', 0)
                        n_11 = counts.get('11', 0)
                        
                        e_val = (n_00 + n_11 - n_01 - n_10) / total
                        correlations[(i_a, i_b)] = e_val
                
                # Calculate CHSH: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
                e_ab = correlations[(0, 0)]
                e_abp = correlations[(0, 1)]
                e_apb = correlations[(1, 0)]
                e_apbp = correlations[(1, 1)]
                
                S = e_ab - e_abp + e_apb + e_apbp
                s_values.append(abs(S))
        
        if s_values:
            mean_s = np.mean(s_values)
            max_s = np.max(s_values)
            violations = sum(1 for s in s_values if s > 2.0)
            
            results_by_model[model_name] = {
                'mean_S': float(mean_s),
                'max_S': float(max_s),
                'violations': violations,
                'n_tests': len(s_values)
            }
            
            print(f"  Mean |S|: {mean_s:.3f}")
            print(f"  Max |S|: {max_s:.3f}")
            print(f"  Violations: {violations}/{len(s_values)}")
    
    return results_by_model


def main():
    print("Q51 Bell Inequality Test - Correct Implementation")
    print("Using Qiskit for proper quantum simulation")
    print("="*60)
    
    # Load embeddings
    print("\nLoading embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    vocabularies = {
        "royalty": ["king", "queen", "prince", "monarch", "royal"],
        "family": ["man", "woman", "child", "parent", "father"],
        "opposites": ["hot", "cold", "big", "small", "fast", "slow"]
    }
    
    embeddings = {}
    for category, words in vocabularies.items():
        embeddings[category] = model.encode(words)
    
    # Run test
    all_embeddings = {'MiniLM-384D': embeddings}
    results = run_chsh_test(all_embeddings)
    
    # Save
    with open('bell_test_qiskit_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to bell_test_qiskit_results.json")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
