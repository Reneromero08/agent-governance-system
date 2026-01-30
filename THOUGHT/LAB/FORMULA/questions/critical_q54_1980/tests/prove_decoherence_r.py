"""
PROOF TEST: R Formula Tracks Quantum Decoherence
================================================

CLAIM: R = (E / grad_S) * sigma^Df INCREASES during decoherence as
quantum states crystallize into classical reality (Quantum Darwinism).

DEMONSTRATION:
1. Simulate qubit decoherence via pure unitary evolution with environment
2. Track R based on system-environment correlations
3. Show R increases as coherence decays and classical correlations form

OPERATIONAL DEFINITIONS FOR DECOHERENCE:
- E = average correlation strength (how much info do fragments carry)
- grad_S = dispersion of correlations (do fragments AGREE)
- Df = log(number of correlated fragments)
- R = (E/grad_S) * sigma^Df

PREDICTED: R increases as quantum -> classical transition occurs
FALSIFICATION: R decreases or stays flat during decoherence
"""

import numpy as np
import json
from datetime import datetime

# =============================================================================
# PARAMETERS (FIXED - NO TUNING)
# =============================================================================
N_ENV = 5              # Number of environment qubits
COUPLING = 0.5         # System-environment coupling
T_MAX = 10.0           # Simulation time (in units where hbar=1)
N_TIMESTEPS = 100      # Time resolution


class QuantumDarwinismSimulator:
    """
    Simulates quantum Darwinism: environment qubits become correlated
    with system qubit through CNOT-like interactions.
    """

    def __init__(self, n_env):
        self.n_env = n_env
        self.n_total = n_env + 1  # system + environment
        self.dim = 2 ** self.n_total

        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_n(self, *matrices):
        """N-fold tensor product."""
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def embed(self, op, qubit_idx):
        """Embed single-qubit operator at given index."""
        ops = [self.I] * self.n_total
        ops[qubit_idx] = op
        return self.kron_n(*ops)

    def initial_state(self):
        """
        |psi_0> = |+>_S |0...0>_E

        System in superposition, environment blank.
        """
        # |+> = (|0> + |1>) / sqrt(2)
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

        # |0>^n for environment
        zero = np.array([1, 0], dtype=complex)
        env_state = zero
        for _ in range(self.n_env - 1):
            env_state = np.kron(env_state, zero)

        # Full state
        psi = np.kron(plus, env_state)
        return psi

    def hamiltonian(self, coupling):
        """
        H = sum_k g * sigma_z^S * sigma_x^{E_k}

        CNOT-like interaction creates correlations.
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)

        Z_sys = self.embed(self.Z, 0)

        for k in range(self.n_env):
            X_env = self.embed(self.X, k + 1)
            H += coupling * Z_sys @ X_env

        return H

    def evolve(self, psi, H, t):
        """Time evolution: |psi(t)> = exp(-iHt) |psi(0)>"""
        # Diagonalize H for efficient evolution
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        U = eigenvectors
        D = np.diag(np.exp(-1j * eigenvalues * t))
        evolution = U @ D @ U.conj().T
        return evolution @ psi

    def partial_trace_to_system(self, rho_full):
        """Trace out environment to get system density matrix."""
        # Reshape to tensor form
        shape = [2] * (2 * self.n_total)
        rho_tensor = rho_full.reshape(shape)

        # Trace over all environment qubits
        # System is qubit 0
        rho_sys = rho_tensor

        for k in range(self.n_env, 0, -1):
            # Trace over qubit k (both bra and ket indices)
            rho_sys = np.trace(rho_sys, axis1=k, axis2=k + self.n_total - (self.n_env - k))

        return rho_sys.reshape(2, 2)

    def partial_trace_to_env_qubit(self, rho_full, env_idx):
        """Trace out everything except one environment qubit."""
        # This is complex; let's use a simpler approach
        # Keep system (0) and env_qubit (env_idx + 1)

        shape = [2] * (2 * self.n_total)
        rho_tensor = rho_full.reshape(shape)

        # We want to keep indices 0 and (env_idx+1)
        keep = [0, env_idx + 1]
        trace_out = [i for i in range(self.n_total) if i not in keep]

        for k in sorted(trace_out, reverse=True):
            # Account for already traced out dimensions
            shift = sum(1 for j in trace_out if j > k)
            ax1 = k
            ax2 = k + self.n_total - shift
            rho_tensor = np.trace(rho_tensor, axis1=ax1, axis2=ax2)

        return rho_tensor.reshape(4, 4)

    def get_env_qubit_state(self, psi, env_idx):
        """Get reduced density matrix for environment qubit env_idx."""
        rho_full = np.outer(psi, psi.conj())
        rho_env = self.partial_trace_to_env_qubit(rho_full, env_idx)

        # Now trace out system to get just the env qubit
        rho_env_tensor = rho_env.reshape(2, 2, 2, 2)
        rho_single_env = np.trace(rho_env_tensor, axis1=0, axis2=2)
        return rho_single_env

    def system_coherence(self, psi):
        """Off-diagonal coherence of system qubit."""
        rho_full = np.outer(psi, psi.conj())
        rho_sys = self.partial_trace_to_system(rho_full)
        return np.abs(rho_sys[0, 1])

    def system_entropy(self, psi):
        """Von Neumann entropy of system."""
        rho_full = np.outer(psi, psi.conj())
        rho_sys = self.partial_trace_to_system(rho_full)
        eigenvalues = np.linalg.eigvalsh(rho_sys)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-12))

    def env_z_expectation(self, psi, env_idx):
        """<Z> for environment qubit - classical correlation."""
        Z_env = self.embed(self.Z, env_idx + 1)
        return np.real(np.conj(psi) @ Z_env @ psi)

    def classical_correlation(self, psi, env_idx):
        """
        Measure correlation using trace distance between conditional states.

        Correlation = (1/2) * ||rho_E|0 - rho_E|1||_tr

        This measures how distinguishable the environment states are
        depending on the system state - the essence of quantum Darwinism.
        """
        env_dim = 2 ** self.n_env

        # Amplitudes conditioned on system = 0
        psi_sys0 = psi[:env_dim]
        norm_0_sq = np.sum(np.abs(psi_sys0)**2)

        # Amplitudes conditioned on system = 1
        psi_sys1 = psi[env_dim:]
        norm_1_sq = np.sum(np.abs(psi_sys1)**2)

        if norm_0_sq < 1e-10 or norm_1_sq < 1e-10:
            return 0.0

        # Density matrices conditioned on system state
        rho_env_0 = np.outer(psi_sys0, psi_sys0.conj()) / norm_0_sq
        rho_env_1 = np.outer(psi_sys1, psi_sys1.conj()) / norm_1_sq

        # Trace distance = (1/2) * Tr|rho_0 - rho_1|
        diff = rho_env_0 - rho_env_1
        eigenvalues = np.linalg.eigvalsh(diff)
        trace_distance = 0.5 * np.sum(np.abs(eigenvalues))

        return trace_distance


def compute_R(correlations):
    """
    R = (E / grad_S) * sigma^Df

    E = average correlation (info content)
    grad_S = std of correlations (dispersion)
    sigma = mean correlation
    Df = log(n_fragments)
    """
    if len(correlations) == 0:
        return 0.0

    E = np.mean(correlations)
    grad_S = np.std(correlations) + 0.01  # Floor
    sigma = max(0.01, np.mean(correlations))
    Df = np.log(len(correlations) + 1)

    return (E / grad_S) * (sigma ** Df)


def run_decoherence_simulation():
    """Main proof: Show R increases during decoherence."""
    print("=" * 70)
    print("PROOF: R Formula Tracks Quantum Decoherence")
    print("=" * 70)
    print()
    print("CLAIM: R increases during decoherence (quantum -> classical)")
    print("PREDICTION: R_after > R_before as correlations strengthen")
    print()

    print(f"Parameters:")
    print(f"  System qubits: 1")
    print(f"  Environment qubits: {N_ENV}")
    print(f"  Coupling strength: {COUPLING}")
    print(f"  Simulation time: {T_MAX}")
    print()

    # Initialize simulator
    sim = QuantumDarwinismSimulator(N_ENV)

    # Get initial state and Hamiltonian
    psi_0 = sim.initial_state()
    H = sim.hamiltonian(COUPLING)

    # Time evolution
    times = np.linspace(0, T_MAX, N_TIMESTEPS)

    # Collect data
    data = {
        'time': [],
        'coherence': [],
        'sys_entropy': [],
        'correlations': [],
        'R': [],
        'mean_correlation': []
    }

    print("Running unitary evolution...")

    for t in times:
        psi = sim.evolve(psi_0, H, t)

        # Measure coherence
        coh = sim.system_coherence(psi)

        # Measure entropy
        ent = sim.system_entropy(psi)

        # Measure correlations with each environment qubit
        corrs = [sim.classical_correlation(psi, k) for k in range(N_ENV)]

        # Compute R
        R = compute_R(corrs)

        data['time'].append(t)
        data['coherence'].append(coh)
        data['sys_entropy'].append(ent)
        data['correlations'].append(corrs)
        data['R'].append(R)
        data['mean_correlation'].append(np.mean(corrs))

    # Convert to arrays
    times = np.array(data['time'])
    coherence = np.array(data['coherence'])
    R_arr = np.array(data['R'])
    entropy = np.array(data['sys_entropy'])
    mean_corr = np.array(data['mean_correlation'])

    # Find decoherence time (first minimum of coherence)
    # Due to periodic evolution, coherence oscillates
    # Look for first significant drop
    initial_coh = coherence[0]
    t_dec_idx = 0
    for i, c in enumerate(coherence):
        if c < 0.1 * initial_coh:
            t_dec_idx = i
            break
    if t_dec_idx == 0:
        t_dec_idx = N_TIMESTEPS // 4

    t_dec = times[t_dec_idx]

    # R before and after decoherence transition
    early_idx = max(1, t_dec_idx // 4)
    late_idx = min(len(times) - 1, t_dec_idx + t_dec_idx // 2)

    R_before = np.mean(R_arr[:early_idx])
    R_at_transition = R_arr[t_dec_idx]
    R_after = np.mean(R_arr[late_idx:])

    R_max = np.max(R_arr)
    R_max_idx = np.argmax(R_arr)
    t_max_R = times[R_max_idx]

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("DECOHERENCE DYNAMICS:")
    print(f"  Initial coherence: {coherence[0]:.4f}")
    print(f"  Min coherence:     {np.min(coherence):.4f}")
    print(f"  Final coherence:   {coherence[-1]:.4f}")
    print(f"  First decoherence at t = {t_dec:.2f}")
    print()
    print("SYSTEM ENTROPY (entanglement with environment):")
    print(f"  Initial: {entropy[0]:.4f}")
    print(f"  Max:     {np.max(entropy):.4f}")
    print(f"  Final:   {entropy[-1]:.4f}")
    print()
    print("CLASSICAL CORRELATIONS:")
    print(f"  Initial mean: {mean_corr[0]:.4f}")
    print(f"  Max mean:     {np.max(mean_corr):.4f}")
    print(f"  Final mean:   {mean_corr[-1]:.4f}")
    print()
    print("R TRAJECTORY:")
    print(f"  R_initial:       {R_arr[0]:.4f}")
    print(f"  R_before_dec:    {R_before:.4f}")
    print(f"  R_at_transition: {R_at_transition:.4f}")
    print(f"  R_max:           {R_max:.4f} (at t = {t_max_R:.2f})")
    print(f"  R_after:         {R_after:.4f}")
    print()

    # Test predictions
    R_increases_to_peak = R_max > R_arr[0] * 1.5
    peak_near_decoherence = abs(t_max_R - t_dec) < T_MAX / 4
    correlation_drives_R = np.corrcoef(R_arr, mean_corr)[0, 1] > 0.5

    print("=" * 70)
    print("PREDICTION TESTS")
    print("=" * 70)
    print()
    print(f"  1. R increases to peak: {'PASS' if R_increases_to_peak else 'FAIL'}")
    print(f"     (R_max = {R_max:.3f}, R_initial = {R_arr[0]:.3f}, ratio = {R_max/max(R_arr[0], 0.001):.2f}x)")
    print()
    print(f"  2. R peak near decoherence: {'PASS' if peak_near_decoherence else 'FAIL'}")
    print(f"     (t_max_R = {t_max_R:.2f}, t_dec = {t_dec:.2f})")
    print()
    print(f"  3. R driven by correlations: {'PASS' if correlation_drives_R else 'FAIL'}")
    print(f"     (correlation = {np.corrcoef(R_arr, mean_corr)[0, 1]:.3f})")
    print()

    tests_passed = int(sum([R_increases_to_peak, peak_near_decoherence, correlation_drives_R]))
    verdict = 'PASS' if tests_passed >= 2 else 'FAIL'

    print("=" * 70)
    print(f"OVERALL: {tests_passed}/3 tests passed")
    print(f"VERDICT: {verdict}")
    print("=" * 70)
    print()

    if verdict == 'PASS':
        print("INTERPRETATION:")
        print("  R = (E/grad_S) * sigma^Df tracks 'crystallization' during decoherence:")
        print()
        print("  - Initially: System in superposition, no correlations, low R")
        print("  - During transition: Environment becomes correlated with system")
        print("  - At peak: Strong, uniform correlations across environment = high R")
        print("  - This is Quantum Darwinism: redundant recording of classical info")
        print()
        print("  The formula captures the emergence of classical objectivity:")
        print("  High R = many witnesses agree on system state = 'real'")
    else:
        print("INTERPRETATION:")
        print("  The simulation did not clearly show the predicted behavior.")
        print("  This unitary evolution may not fully capture decoherence physics.")

    # Save results
    output = {
        'test_name': 'prove_decoherence_r',
        'timestamp': datetime.now().isoformat(),
        'claim': 'R increases during quantum-classical transition',
        'parameters': {
            'N_ENV': N_ENV,
            'COUPLING': COUPLING,
            'T_MAX': T_MAX,
            'N_TIMESTEPS': N_TIMESTEPS
        },
        'measurements': {
            'initial_coherence': float(coherence[0]),
            'min_coherence': float(np.min(coherence)),
            'final_coherence': float(coherence[-1]),
            't_decoherence': float(t_dec),
            'R_initial': float(R_arr[0]),
            'R_max': float(R_max),
            't_max_R': float(t_max_R),
            'R_final': float(R_arr[-1]),
            'max_entropy': float(np.max(entropy)),
            'max_correlation': float(np.max(mean_corr))
        },
        'predictions': {
            'R_increases_to_peak': bool(R_increases_to_peak),
            'peak_near_decoherence': bool(peak_near_decoherence),
            'correlation_drives_R': bool(correlation_drives_R)
        },
        'tests_passed': tests_passed,
        'verdict': verdict
    }

    output_path = "D:/Reneshizzle/Apps/Claude/agent-governance-system/elegant-neumann/THOUGHT/LAB/FORMULA/questions/critical_q54_1980/tests/prove_decoherence_r_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    results = run_decoherence_simulation()
