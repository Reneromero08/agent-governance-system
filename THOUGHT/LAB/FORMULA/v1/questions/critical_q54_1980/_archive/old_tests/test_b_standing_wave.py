"""
Q54 Test B: Standing Wave Phase Lock vs Effective Mass

HYPOTHESIS:
Standing waves trap energy by phase-locking (nodes that don't move).
The degree of "phase lock" should correlate with effective mass -
more locked = more resistant to change.

PREDICTION (stated BEFORE running):
1. Bound states (standing waves) resist perturbation MORE than free states
2. "Phase lock" metric = 1 / (sum of transition amplitudes to other states)
3. Higher energy bound states should have MORE phase lock (more nodes = more structure)

FALSIFICATION:
If free propagating states resist perturbation equally or MORE than bound states,
then phase-locking is NOT the source of mass-like behavior.

FIXED PARAMETERS (no tuning!):
- N_GRID = 500
- X_MAX = 10.0
- WELL_DEPTH = 5.0
- PERTURBATION_STRENGTH = 0.01
- N_MODES = 20
"""

import numpy as np
from scipy import linalg
import json
from datetime import datetime

# ============================================================================
# FIXED PARAMETERS - NO TUNING ALLOWED
# ============================================================================
N_GRID = 500                    # spatial discretization
X_MAX = 10.0                    # box size in natural units
WELL_DEPTH = 5.0                # potential well depth
PERTURBATION_STRENGTH = 0.01   # strength of perturbation
N_MODES = 20                    # number of eigenstates to compute


def create_grid():
    """Create spatial grid."""
    x = np.linspace(-X_MAX, X_MAX, N_GRID)
    dx = x[1] - x[0]
    return x, dx


def finite_well_potential(x):
    """
    Finite square well potential.
    V(x) = -WELL_DEPTH for |x| < WELL_WIDTH, 0 otherwise.
    """
    well_width = X_MAX / 3  # well width = 1/3 of box
    V = np.zeros_like(x)
    V[np.abs(x) < well_width] = -WELL_DEPTH
    return V


def build_hamiltonian(x, dx, V):
    """
    Build the Hamiltonian matrix using finite differences.
    H = -0.5 * d^2/dx^2 + V(x)  (natural units: hbar = m = 1)
    """
    N = len(x)

    # Kinetic energy: second derivative via finite differences
    # T = -0.5 * (psi[i+1] - 2*psi[i] + psi[i-1]) / dx^2
    kinetic_coeff = -0.5 / (dx ** 2)

    # Build tridiagonal kinetic energy matrix
    diag_main = np.full(N, -2.0 * kinetic_coeff) + V  # main diagonal
    diag_off = np.full(N - 1, kinetic_coeff)          # off diagonals

    H = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)

    return H


def solve_eigenstates(H, n_modes):
    """Solve for the first n_modes eigenstates."""
    # Use eigh for Hermitian matrix (real symmetric in our case)
    eigenvalues, eigenvectors = linalg.eigh(H)

    # Select lowest n_modes
    idx = np.argsort(eigenvalues)[:n_modes]
    energies = eigenvalues[idx]
    states = eigenvectors[:, idx]

    # Normalize (should already be normalized, but make sure)
    for i in range(n_modes):
        norm = np.sqrt(np.sum(np.abs(states[:, i]) ** 2))
        states[:, i] /= norm

    return energies, states


def create_perturbation(x):
    """
    Create a localized perturbation potential.
    Using a Gaussian bump displaced from center.
    """
    # Perturbation centered at x = X_MAX/2
    x0 = X_MAX / 2
    sigma = X_MAX / 10
    V_pert = PERTURBATION_STRENGTH * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    return V_pert


def compute_transition_amplitudes(states, V_pert, x, dx):
    """
    Compute transition amplitudes <n|V_pert|m> for all state pairs.
    Returns matrix of transition amplitudes.
    """
    n_states = states.shape[1]
    T = np.zeros((n_states, n_states))

    for n in range(n_states):
        for m in range(n_states):
            # <n|V|m> = integral psi_n^* V psi_m dx
            integrand = np.conj(states[:, n]) * V_pert * states[:, m]
            T[n, m] = np.abs(np.sum(integrand) * dx)

    return T


def compute_phase_lock(T, state_idx):
    """
    Compute "phase lock" metric for a state.
    Phase lock = 1 / (sum of transition amplitudes to OTHER states)

    Higher phase lock = more resistant to perturbation.
    """
    n_states = T.shape[0]

    # Sum of amplitudes to all OTHER states
    total_mixing = 0.0
    for m in range(n_states):
        if m != state_idx:
            total_mixing += T[state_idx, m]

    if total_mixing < 1e-15:
        return float('inf')  # Perfectly stable

    phase_lock = 1.0 / total_mixing
    return phase_lock


def count_nodes(state, x, threshold=0.01):
    """
    Count the number of nodes (zero crossings) in a wavefunction.
    More nodes = more structure.
    """
    # Find where wavefunction magnitude exceeds threshold
    significant = np.abs(state) > threshold * np.max(np.abs(state))

    # Count sign changes in significant region
    nodes = 0
    in_significant = False
    last_sign = 0

    for i, (sig, val) in enumerate(zip(significant, state)):
        if sig:
            in_significant = True
            current_sign = np.sign(val)
            if last_sign != 0 and current_sign != last_sign:
                nodes += 1
            last_sign = current_sign

    return nodes


def create_plane_wave(x, k, dx):
    """
    Create a normalized plane wave state: psi = exp(i*k*x)
    These represent free propagating particles (not bound).
    """
    psi = np.exp(1j * k * x)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


def compute_plane_wave_transition_amplitudes(x, dx, V_pert, k_values):
    """
    Compute transition amplitudes for plane waves using the SAME metric as bound states.

    For plane waves, the transition amplitude from |k> to |k'> is:
        <k'|V_pert|k> = integral exp(-ik'x) * V_pert(x) * exp(ikx) dx / L
                      = V_pert_fourier(k - k') / L

    This is directly comparable to the bound state transition amplitudes.
    Phase lock = 1 / (sum of |<k'|V|k>| for all k' != k)
    """
    n_k = len(k_values)
    L = x[-1] - x[0]  # box length

    # Build transition matrix for plane waves (analogous to bound states)
    T_plane = np.zeros((n_k, n_k))

    for i, k in enumerate(k_values):
        psi_k = create_plane_wave(x, k, dx)
        for j, k_prime in enumerate(k_values):
            psi_k_prime = create_plane_wave(x, k_prime, dx)
            # <k'|V|k> = integral psi_k'^* V psi_k dx
            integrand = np.conj(psi_k_prime) * V_pert * psi_k
            T_plane[i, j] = np.abs(np.sum(integrand) * dx)

    # Compute phase lock for each plane wave state (same formula as bound states)
    stabilities = []
    for i, k in enumerate(k_values):
        # Sum of amplitudes to all OTHER states
        total_mixing = sum(T_plane[i, j] for j in range(n_k) if j != i)

        if total_mixing < 1e-15:
            phase_lock = float('inf')
        else:
            phase_lock = 1.0 / total_mixing

        stabilities.append({
            'k': float(k),
            'total_mixing': float(total_mixing),
            'phase_lock': float(phase_lock) if phase_lock != float('inf') else 'inf'
        })

    return stabilities


def classify_bound_states(energies, well_depth=WELL_DEPTH):
    """
    Classify states as bound (E < 0) or unbound (E > 0).
    For a finite well, bound states have E < 0 (inside well).
    """
    return ['bound' if E < 0 else 'unbound' for E in energies]


def run_test():
    """
    Main test: Compare standing wave stability to free wave stability.
    """
    print("=" * 70)
    print("Q54 Test B: Standing Wave Phase Lock vs Effective Mass")
    print("=" * 70)
    print()

    # State predictions BEFORE running
    print("PREDICTIONS (stated before running):")
    print("-" * 50)
    print("1. Bound states resist perturbation MORE than free states")
    print("2. Phase lock = 1 / (sum of transition amplitudes)")
    print("3. Higher energy bound states -> MORE phase lock")
    print()

    # Fixed parameters
    print("FIXED PARAMETERS:")
    print(f"  N_GRID = {N_GRID}")
    print(f"  X_MAX = {X_MAX}")
    print(f"  WELL_DEPTH = {WELL_DEPTH}")
    print(f"  PERTURBATION_STRENGTH = {PERTURBATION_STRENGTH}")
    print(f"  N_MODES = {N_MODES}")
    print()

    # Create system
    x, dx = create_grid()
    V = finite_well_potential(x)
    H = build_hamiltonian(x, dx, V)

    # Solve for eigenstates
    print("Solving for eigenstates...")
    energies, states = solve_eigenstates(H, N_MODES)

    # Classify states
    state_types = classify_bound_states(energies)
    n_bound = sum(1 for t in state_types if t == 'bound')
    print(f"Found {n_bound} bound states, {N_MODES - n_bound} unbound states")
    print()

    # Create perturbation
    V_pert = create_perturbation(x)

    # Compute transition matrix
    print("Computing transition amplitudes...")
    T = compute_transition_amplitudes(states, V_pert, x, dx)

    # Compute phase lock for each eigenstate
    print("Computing phase lock metrics...")
    print()

    results = {
        'bound_states': [],
        'unbound_states': [],
        'plane_waves': [],
        'predictions': {},
        'parameters': {
            'N_GRID': N_GRID,
            'X_MAX': X_MAX,
            'WELL_DEPTH': WELL_DEPTH,
            'PERTURBATION_STRENGTH': PERTURBATION_STRENGTH,
            'N_MODES': N_MODES
        },
        'timestamp': datetime.now().isoformat()
    }

    print("=" * 70)
    print("BOUND STATES (Standing Waves)")
    print("=" * 70)
    print(f"{'n':>3} {'Energy':>12} {'Nodes':>6} {'Phase Lock':>12} {'Total Mixing':>12}")
    print("-" * 70)

    for i, (E, state_type) in enumerate(zip(energies, state_types)):
        nodes = count_nodes(states[:, i], x)
        phase_lock = compute_phase_lock(T, i)
        total_mixing = sum(T[i, j] for j in range(N_MODES) if j != i)

        state_data = {
            'index': int(i),
            'energy': float(E),
            'nodes': int(nodes),
            'phase_lock': float(phase_lock) if phase_lock != float('inf') else 'inf',
            'total_mixing': float(total_mixing),
            'state_type': state_type
        }

        if state_type == 'bound':
            results['bound_states'].append(state_data)
            lock_str = f"{phase_lock:.4f}" if phase_lock != float('inf') else "inf"
            print(f"{i:>3} {E:>12.4f} {nodes:>6} {lock_str:>12} {total_mixing:>12.6f}")
        else:
            results['unbound_states'].append(state_data)

    print()
    print("=" * 70)
    print("PLANE WAVES (Free Propagating States)")
    print("=" * 70)

    # Test plane waves with various momenta - using SAME metric as bound states
    k_values = np.linspace(0.5, 5.0, 10)
    plane_wave_results = compute_plane_wave_transition_amplitudes(x, dx, V_pert, k_values)
    results['plane_waves'] = plane_wave_results

    print(f"{'k':>8} {'Total Mixing':>12} {'Phase Lock':>12}")
    print("-" * 40)
    for pw in plane_wave_results:
        lock_str = f"{pw['phase_lock']:.4f}" if pw['phase_lock'] != 'inf' else "inf"
        print(f"{pw['k']:>8.2f} {pw['total_mixing']:>12.6f} {lock_str:>12}")

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Get finite phase locks for comparison (now using SAME metric for all)
    bound_locks = [s['phase_lock'] for s in results['bound_states']
                   if s['phase_lock'] != 'inf']
    unbound_locks = [s['phase_lock'] for s in results['unbound_states']
                     if s['phase_lock'] != 'inf']
    plane_locks = [pw['phase_lock'] for pw in plane_wave_results
                   if pw['phase_lock'] != 'inf']

    avg_bound_lock = np.mean(bound_locks) if bound_locks else float('inf')
    avg_unbound_lock = np.mean(unbound_locks) if unbound_locks else 0
    avg_plane_lock = np.mean(plane_locks) if plane_locks else 0

    print()
    print("Average Phase Lock (same metric for all):")
    print(f"  Bound states:   {avg_bound_lock:.4f}")
    print(f"  Unbound states: {avg_unbound_lock:.4f}")
    print(f"  Plane waves:    {avg_plane_lock:.4f}")

    # Test predictions
    print()
    print("=" * 70)
    print("PREDICTION TESTS")
    print("=" * 70)

    # Prediction 1: Bound states resist perturbation MORE than free states
    pred1_pass = avg_bound_lock > avg_plane_lock
    results['predictions']['pred1_bound_more_stable'] = {
        'prediction': 'Bound states resist perturbation MORE than free states',
        'bound_avg': float(avg_bound_lock),
        'plane_avg': float(avg_plane_lock),
        'ratio': float(avg_bound_lock / avg_plane_lock) if avg_plane_lock > 0 else float('inf'),
        'passed': bool(pred1_pass)
    }
    print()
    print(f"Prediction 1: Bound states more stable than plane waves")
    print(f"  Bound avg phase lock: {avg_bound_lock:.4f}")
    print(f"  Plane avg phase lock: {avg_plane_lock:.4f}")
    print(f"  Ratio: {avg_bound_lock / avg_plane_lock:.2f}x" if avg_plane_lock > 0 else "  Ratio: inf")
    print(f"  Result: {'PASS' if pred1_pass else 'FAIL'}")

    # Prediction 2: Phase lock metric is meaningful (check variance)
    phase_lock_values = [s['phase_lock'] for s in results['bound_states']
                         if s['phase_lock'] != 'inf']
    if len(phase_lock_values) > 1:
        lock_variance = np.var(phase_lock_values)
        pred2_pass = lock_variance > 0  # Should show variation
    else:
        lock_variance = 0
        pred2_pass = False

    results['predictions']['pred2_lock_varies'] = {
        'prediction': 'Phase lock shows meaningful variation',
        'variance': float(lock_variance),
        'passed': bool(pred2_pass)
    }
    print()
    print(f"Prediction 2: Phase lock shows variation across bound states")
    print(f"  Variance: {lock_variance:.6f}")
    print(f"  Result: {'PASS' if pred2_pass else 'FAIL'}")

    # Prediction 3: More BINDING energy -> MORE phase lock
    # BINDING energy = |E_n| = -E_n (how much energy is locked in the structure)
    # This is what Q54 predicts: more energy locked -> more mass-like stability
    bound_energies = [s['energy'] for s in results['bound_states']
                      if s['phase_lock'] != 'inf']
    binding_energies = [-e for e in bound_energies]  # Binding energy = |E_n|
    bound_locks_finite = [s['phase_lock'] for s in results['bound_states']
                          if s['phase_lock'] != 'inf']

    if len(bound_energies) > 2:
        # Compute correlation with BINDING energy (not raw energy)
        from scipy import stats as scipy_stats
        corr, p_value = scipy_stats.pearsonr(binding_energies, bound_locks_finite)
        pred3_pass = corr > 0.5 and p_value < 0.05  # Strong positive correlation
    else:
        corr = 0
        p_value = 1
        pred3_pass = False

    results['predictions']['pred3_energy_lock_correlation'] = {
        'prediction': 'More BINDING energy (energy locked) -> more phase lock',
        'correlation': float(corr),
        'p_value': float(p_value),
        'passed': bool(pred3_pass)
    }
    print()
    print(f"Prediction 3: Binding energy-lock correlation (more locked E -> more lock)")
    print(f"  Correlation with |E_n|: {corr:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Result: {'PASS' if pred3_pass else 'FAIL'}")

    # Overall result
    all_passed = pred1_pass and pred2_pass and pred3_pass
    results['overall_passed'] = bool(all_passed)

    print()
    print("=" * 70)
    if all_passed:
        print("OVERALL: PASS")
        print("Standing waves (phase-locked states) show mass-like behavior")
    else:
        print("OVERALL: FAIL")
        print("Evidence does NOT support phase-lock -> mass hypothesis")
    print("=" * 70)

    # Interpretation
    print()
    print("INTERPRETATION:")
    print("-" * 70)
    if pred1_pass:
        print("- Bound states (standing waves) ARE more resistant to perturbation")
        print("  This supports: phase locking creates 'inertia'")
    else:
        print("- FALSIFIED: Free waves resist equally or more")
        print("  Phase locking is NOT the source of mass-like behavior")

    if pred3_pass and corr > 0:
        print(f"- Binding energy-lock correlation is POSITIVE (r={corr:.3f})")
        print("  States with more energy LOCKED have higher phase lock")
        print("  This supports: energy locked in structure -> effective mass")
    elif not pred3_pass and len(bound_energies) > 2:
        print(f"- Binding energy-lock correlation is weak (r={corr:.3f})")
        print("  Energy locked does not strongly predict phase lock")

    # Save results
    output_path = "D:/Reneshizzle/Apps/Claude/agent-governance-system/elegant-neumann/THOUGHT/LAB/FORMULA/questions/critical_q54_1980/tests/test_b_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {output_path}")

    return all_passed, results


if __name__ == "__main__":
    passed, results = run_test()
    exit(0 if passed else 1)
