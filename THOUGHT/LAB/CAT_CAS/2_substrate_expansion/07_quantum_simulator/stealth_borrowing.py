"""
Stealth-Borrowing Entanglement Test (Grail 1)

Verifies that a qubit in a highly entangled state can be borrowed as a
catalytic tape, perform a quantum computation using a clean qubit, and be
restored to its original state (preserving the entanglement with an external
untouched reference qubit) without collapsing the wave function.

Integrates Semiotic phase-rotation math from FORMULA/v4/SEMIOTIC_LIGHT_CONE_1_1:
  U_σ rotates the state by θ = π * (1 - 1 / (σ(f)^D_f * E / ∇S))
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import qiskit.quantum_info as qi

def calculate_chsh(density_matrix_01):
    """
    Computes the maximum CHSH correlation value for a 2-qubit density matrix.
    For a maximally entangled Bell state, this is 2 * sqrt(2) ≈ 2.828.
    For a classical mixed state, this is <= 2.0.
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # We construct the correlation matrix T where T_ij = Tr(rho * (sigma_i \otimes \sigma_j))
    T = np.zeros((3, 3))
    paulis = [X, Y, Z]
    for i in range(3):
        for j in range(3):
            op = np.kron(paulis[i], paulis[j])
            T[i, j] = np.real(np.trace(density_matrix_01 @ op))
            
    # The maximum CHSH violation is 2 * sqrt(s_1^2 + s_2^2) where s_1, s_2 are the two largest singular values of T
    singular_values = np.linalg.svd(T, compute_uv=False)
    sorted_sv = np.sort(singular_values)[::-1]
    chsh = 2 * np.sqrt(sorted_sv[0]**2 + sorted_sv[1]**2)
    return chsh

def run_stealth_borrowing_experiment(theta=np.pi/3, ablate=False):
    """
    Executes the stealth-borrowing experiment.
    Q0: Reference Qubit (never touched by algorithm)
    Q1: Catalyst/Dirty Tape Qubit (borrowed, used, and restored)
    Q2: Clean Qubit (initialized to |0>, target of computation)
    """
    rng = np.random.default_rng(seed=42)
    print(f"\n--- Running Experiment (theta = {theta:.3f} rad, ablate = {ablate}) ---")
    
    # We will use qiskit.quantum_info to build and trace states at each step
    
    # 1. Prepare initial state: Q0 and Q1 in Bell State |Phi^+>, Q2 in |0>
    # |psi_0> = (|000> + |110>) / sqrt(2)
    init_circuit = QuantumCircuit(3)
    init_circuit.h(0)
    init_circuit.cx(0, 1)
    state_0 = qi.Statevector.from_instruction(init_circuit)
    
    # Measure initial entanglement metrics
    rho_01_step0 = qi.partial_trace(state_0, [2]).data
    chsh_step0 = calculate_chsh(rho_01_step0)
    bell_state = qi.Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    fidelity_step0 = qi.state_fidelity(rho_01_step0, bell_state)
    
    print(f"Step 0: Initial Bell State prepared on (Q0, Q1)")
    print(f"  - (Q0, Q1) CHSH value: {chsh_step0:.4f} (Quantum limit: 2.828)")
    print(f"  - Fidelity with |Phi^+>: {fidelity_step0:.4f}")
    
    # 2. Start Catalytic Computation
    # Step 1: CX(Q1, Q2) to entangle the clean qubit with the tape
    step1_circuit = init_circuit.copy()
    step1_circuit.cx(1, 2)
    state_1 = qi.Statevector.from_instruction(step1_circuit)
    
    rho_01_step1 = qi.partial_trace(state_1, [2]).data
    chsh_step1 = calculate_chsh(rho_01_step1)
    fidelity_step1 = qi.state_fidelity(rho_01_step1, bell_state)
    
    print(f"Step 1: Q1 entangled with clean Q2 (GHZ state)")
    print(f"  - (Q0, Q1) CHSH value: {chsh_step1:.4f} (Classical limit: <= 2.0)")
    print(f"  - Fidelity with |Phi^+>: {fidelity_step1:.4f} (Entanglement is 'borrowed')")
    
    # Step 2: Apply target computation on Q2 (Rx(theta) representing Semiotic phase rotation)
    step2_circuit = step1_circuit.copy()
    step2_circuit.rx(theta, 2)
    
    if ablate:
        # Ablation mode: measure Q1 mid-computation (representing an external probe/erasure)
        # This collapses the statevector
        state_temp = qi.Statevector.from_instruction(step2_circuit)
        # Simulate measurement on qubit 1
        prob_0 = np.real(state_temp.probabilities([1])[0])
        measured_val = 0 if rng.random() < prob_0 else 1
        # Project state vector
        projector = np.zeros((8, 8))
        for i in range(8):
            # check if bit 1 matches measured_val
            if ((i >> 1) & 1) == measured_val:
                projector[i, i] = 1.0
        # Normalize
        raw_state = state_temp.data
        projected = projector @ raw_state
        projected = projected / np.linalg.norm(projected)
        state_2 = qi.Statevector(projected)
        print(f"Step 2 (Ablated): Measured Q1 mid-run to value {measured_val} (collapsed wavefunction!)")
        
        # Re-apply the circuit step 3 with the collapsed state
        step3_circuit = QuantumCircuit(3)
        # We manually initialize to the collapsed state and apply final CX
        step3_circuit.initialize(state_2.data, range(3))
        step3_circuit.cx(1, 2)
        state_3 = qi.Statevector.from_instruction(step3_circuit)
    else:
        # Normal mode: perform target computation without collapsing the tape
        state_2 = qi.Statevector.from_instruction(step2_circuit)
        
        # Step 3: CX(Q1, Q2) to restore the tape and unentangle the clean qubit
        step3_circuit = step2_circuit.copy()
        step3_circuit.cx(1, 2)
        state_3 = qi.Statevector.from_instruction(step3_circuit)
        print(f"Step 2: Applied Semiotic rotation Rx({theta:.3f}) on clean Q2")
        print(f"Step 3: Restored Q1 tape via reverse CX(Q1, Q2)")
        
    # Measure final entanglement metrics
    rho_01_step3 = qi.partial_trace(state_3, [2]).data
    chsh_step3 = calculate_chsh(rho_01_step3)
    fidelity_step3 = qi.state_fidelity(rho_01_step3, bell_state)
    
    # Measure final state of clean qubit Q2
    rho_2 = qi.partial_trace(state_3, [0, 1]).data
    # Theoretical target state: cos(theta/2)|0> - i * sin(theta/2)|1>
    target_q2_state = qi.Statevector([np.cos(theta/2), -1j * np.sin(theta/2)])
    fidelity_q2 = qi.state_fidelity(rho_2, target_q2_state)
    
    print(f"Final Results:")
    print(f"  - (Q0, Q1) CHSH value: {chsh_step3:.4f} (Goal: 2.828)")
    print(f"  - Fidelity with |Phi^+>: {fidelity_step3:.4f} (Goal: 1.000)")
    print(f"  - Clean Q2 State Fidelity with target: {fidelity_q2:.4f} (Goal: 1.000)")
    
    return chsh_step3, fidelity_step3, fidelity_q2

def main():
    print("=" * 70)
    print("GRAIL 1: QUANTUM STEALTH-BORROWING ENTANGLEMENT TEST")
    print("=" * 70)
    
    # Define Semiotic parameters to calculate theta
    E = 1.0       # Essence
    nabla_S = 0.5 # Entropy gradient
    sigma_f = 1.5 # Constitutional compression
    D_f = 2       # Redundancy depth
    
    # Semiotic Formula rotation angle
    # theta = pi * (1 - 1 / (sigma_f^D_f * E / nabla_S))
    semiotic_amplification = (sigma_f ** D_f) * (E / nabla_S)
    theta = np.pi * (1.0 - 1.0 / semiotic_amplification)
    
    print(f"[Semiotic Setup]")
    print(f"  - Essence (E):                  {E}")
    print(f"  - Entropy Gradient (nabla_S):   {nabla_S}")
    print(f"  - Compression (sigma):          {sigma_f}")
    print(f"  - Redundancy (D_f):             {D_f}")
    print(f"  - Semiotic Amplification:       {semiotic_amplification:.3f}")
    print(f"  - Calculated Phase Angle (theta): {theta:.3f} rad ({np.degrees(theta):.1f} deg)")
    
    # Run Normal Catalytic Run
    chsh_norm, fid_norm, fid_q2_norm = run_stealth_borrowing_experiment(theta=theta, ablate=False)
    
    # Run Ablated Run (Measurement/collapsing during computation)
    chsh_ab, fid_ab, fid_q2_ab = run_stealth_borrowing_experiment(theta=theta, ablate=True)
    
    print("\n" + "=" * 70)
    print("VERDICT & SUMMARY")
    print("=" * 70)
    print(f"Normal Run CHSH:  {chsh_norm:.4f} (Fidelity: {fid_norm:.2%})")
    print(f"Ablated Run CHSH: {chsh_ab:.4f} (Fidelity: {fid_ab:.2%})")
    
    success = (chsh_norm > 2.8) and (fid_norm > 0.99) and (fid_q2_norm > 0.99) and (chsh_ab <= 2.1)
    if success:
        print("\n[VERDICT: SUCCESS]")
        print("  Entanglement stealth-borrowing successfully verified!")
        print("  1. Entanglement remains 100% intact when run unitarily.")
        print("  2. Computation on Q2 executed with 100% fidelity.")
        print("  3. Measuring/erasing Q1 mid-run collapses entanglement permanently.")
    else:
        print("\n[VERDICT: FAILED]")
        print("  Verification criteria not met.")
        
if __name__ == "__main__":
    main()
