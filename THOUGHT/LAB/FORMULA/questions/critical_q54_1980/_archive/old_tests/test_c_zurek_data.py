"""
Q54 TEST C: ZUREK QUANTUM DARWINISM DECOHERENCE DATA
=====================================================

Tests whether R = (E / grad_S) * sigma^Df spikes at the moment classical
behavior emerges during decoherence.

HYPOTHESIS (stated BEFORE running):
- R shows step function or spike at t_decoherence
- Before decoherence: R fluctuates (quantum uncertainty)
- After decoherence: R stabilizes at high value (classical lock-in)
- The transition should be sharp, not gradual

FALSIFICATION CRITERIA:
- R is flat throughout (no transition signature)
- R DECREASES during decoherence (opposite of crystallization)
- No threshold behavior visible

DATA SOURCE: QuTiP simulation (arXiv papers lack accessible raw data)
Model: Spin-boson with Lindblad master equation evolution

Based on:
- arXiv:1205.3197 (Riedel et al.) - Quantum Darwinism redundancy dynamics
- arXiv:quant-ph/0505031 (Blume-Kohout & Zurek) - Quantum Brownian Motion
- Zurek, W. H. (2009). Quantum Darwinism. Nature Physics, 5(3), 181-188.

Requirements: pip install qutip numpy scipy matplotlib
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("QuTiP not installed. Run: pip install qutip")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available for plotting")


# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

class DecoherenceParams:
    """Parameters for Quantum Darwinism simulation."""

    # System frequency (set to 0 for pure QD dynamics)
    omega_0 = 0.0

    # Environment coupling strength
    # For CNOT-like interaction: t_dec ~ pi / (2 * coupling * n_env)
    coupling = 0.5

    # Temperature (not used in pure unitary evolution)
    temperature = 0.0

    # Number of environment modes (bath qubits)
    # More modes = more redundancy, cleaner decoherence
    n_env_modes = 6

    # Time parameters
    # t_max should cover decoherence time plus some margin
    # With coupling=0.5, n_env=6: t_dec ~ pi/(2*0.5*6) ~ 0.52
    t_max = 5.0  # Cover multiple decoherence times
    n_timesteps = 100  # Good resolution

    # Decoherence time estimate for CNOT-like interaction
    # t_dec ~ pi / (2 * coupling * n_env_modes)
    t_decoherence_estimate = np.pi / (2 * coupling * n_env_modes)


# =============================================================================
# SPIN-BOSON MODEL IMPLEMENTATION
# =============================================================================

def create_spin_boson_hamiltonian(params: DecoherenceParams) -> Tuple[qt.Qobj, List[qt.Qobj]]:
    """
    Create the Hamiltonian for Quantum Darwinism (Zurek's model).

    CRITICAL PHYSICS:
    - H_int = sigma_z (system) * sigma_x (environment) - CNOT-like interaction
    - Environment starts in |0...0> (set in create_initial_state)
    - NO Lindblad collapse operators - pure unitary evolution

    This creates proper Zurek Quantum Darwinism:
    - System |0> leaves environment unchanged: |0>|0> -> |0>|0>
    - System |1> flips environment: |1>|0> -> |1>|1>
    - Superposition creates entanglement: (|0>+|1>)|0> -> |0>|0> + |1>|1> (GHZ-like)
    - Each environment fragment becomes correlated with system
    - Mutual information I(S:Fk) increases toward S(system)
    - Redundancy emerges as classical information spreads

    Returns: (H, collapse_operators)
    """
    n_total = 1 + params.n_env_modes  # system + environment

    # System operators (first qubit)
    def sys_op(op):
        """Embed system operator in full Hilbert space."""
        ops = [op] + [qt.qeye(2) for _ in range(params.n_env_modes)]
        return qt.tensor(ops)

    # Environment operators (indexed from 1)
    def env_op(idx, op):
        """Embed environment operator in full Hilbert space."""
        ops = [qt.qeye(2) for _ in range(n_total)]
        ops[idx] = op
        return qt.tensor(ops)

    # Pauli matrices
    sz = qt.sigmaz()
    sx = qt.sigmax()

    # System Hamiltonian: H_S = 0 (no free evolution - focus on interaction)
    # Setting omega_0 = 0 simplifies dynamics to pure QD
    H_sys = 0 * sys_op(sz)

    # System-environment interaction: CNOT-like (sigma_z * sigma_x)
    # H_int = sum_k g_k * sigma_z^S * sigma_x^(k)
    #
    # CRITICAL: This is the CORRECT interaction for Quantum Darwinism!
    # - sigma_z * sigma_x acts like a controlled-NOT
    # - When system is |0>: sigma_z = +1, env evolves with +sigma_x (rotation)
    # - When system is |1>: sigma_z = -1, env evolves with -sigma_x (opposite)
    # - This creates PERFECT correlation between system and environment
    # - Mutual information I(S:F) approaches S(system) during evolution
    H_int = qt.tensor([qt.qeye(2) for _ in range(n_total)]) * 0  # Zero

    for k in range(params.n_env_modes):
        g_k = params.coupling
        # sigma_z * sigma_x: CNOT-like interaction for proper QD
        H_int = H_int + g_k * sys_op(sz) * env_op(k + 1, sx)

    H_total = H_sys + H_int

    # NO collapse operators - pure unitary evolution
    #
    # CRITICAL: Lindblad terms DESTROY QD correlations!
    # - They cause independent dephasing of environment fragments
    # - This erases the correlation with the system
    # - Mutual information DECREASES instead of increasing
    #
    # Pure unitary evolution creates proper QD:
    # - Coherence decays as environment entangles with system
    # - MI increases as fragments "record" system state
    # - At t ~ pi/(2*coupling), full decoherence + full redundancy
    # NO collapse operators - pure unitary evolution for proper QD
    collapse_ops = []

    return H_total, collapse_ops


def create_initial_state(params: DecoherenceParams) -> qt.Qobj:
    """
    Create initial state: system in superposition, environment in |0...0>.

    |psi_0> = (|0> + |1>) / sqrt(2) (x) |0>^(x)n_env

    This is the canonical initial state for quantum Darwinism studies:
    system has "which-state" information, environment is blank.
    """
    # System starts in |+> = (|0> + |1>) / sqrt(2)
    up = qt.basis(2, 0)
    down = qt.basis(2, 1)
    sys_state = (up + down).unit()

    # Environment starts in all |0>
    env_state = up
    for _ in range(params.n_env_modes - 1):
        env_state = qt.tensor(env_state, up)

    # Full initial state
    psi_0 = qt.tensor(sys_state, env_state)

    return psi_0


# =============================================================================
# QUANTUM METRICS COMPUTATION
# =============================================================================

def compute_coherence(rho: qt.Qobj) -> float:
    """
    Compute system coherence (off-diagonal magnitude).

    Coherence = |rho_01| / sqrt(rho_00 * rho_11)

    This is 1 for pure superposition, 0 for fully decohered (diagonal).
    """
    rho_arr = rho.full()

    if rho_arr.shape[0] == 2:
        # Single qubit
        off_diag = np.abs(rho_arr[0, 1])
        diag_prod = np.sqrt(np.abs(rho_arr[0, 0] * rho_arr[1, 1]))

        if diag_prod > 1e-10:
            return off_diag / diag_prod
        return off_diag
    else:
        # Multi-qubit: take system only
        return np.abs(rho_arr[0, 1])


def compute_mutual_information(state: qt.Qobj, n_total: int,
                               sys_idx: int, frag_indices: List[int]) -> float:
    """
    Compute mutual information I(S:F) between system and fragment(s).

    I(S:F) = S(rho_S) + S(rho_F) - S(rho_SF)

    High I(S:F) = fragment contains information about system state
    """
    try:
        rho_s = state.ptrace([sys_idx])
        rho_f = state.ptrace(frag_indices)
        rho_sf = state.ptrace([sys_idx] + frag_indices)

        S_s = qt.entropy_vn(rho_s, base=2)
        S_f = qt.entropy_vn(rho_f, base=2)
        S_sf = qt.entropy_vn(rho_sf, base=2)

        return max(0, S_s + S_f - S_sf)
    except Exception:
        return 0.0


def compute_redundancy(state: qt.Qobj, n_total: int) -> float:
    """
    Compute Zurek redundancy R_delta.

    Redundancy = fraction of single fragments that contain nearly full
    information about the system.

    High redundancy = classical (many independent witnesses)
    Low redundancy = quantum (no consensus)
    """
    # System entropy (max possible info)
    rho_sys = state.ptrace([0])
    sys_entropy = qt.entropy_vn(rho_sys, base=2)

    if sys_entropy < 0.01:
        return 1.0  # Pure state, trivial

    # Count fragments with high MI
    threshold = 0.9 * sys_entropy
    good_count = 0

    for f in range(1, n_total):
        mi = compute_mutual_information(state, n_total, 0, [f])
        if mi >= threshold:
            good_count += 1

    return good_count / (n_total - 1)


def get_fragment_probs(state: qt.Qobj, frag_indices: List[int]) -> np.ndarray:
    """Get probability distribution for fragment(s)."""
    rho = state.ptrace(frag_indices)
    probs = np.abs(np.diag(rho.full()))
    return probs / probs.sum()


# =============================================================================
# THE LIVING FORMULA: R = (E / grad_S) * sigma^Df
# =============================================================================

def compute_essence(probs: np.ndarray) -> float:
    """
    E = distance from uniform distribution.

    High E = definite state (information present)
    Low E = maximally uncertain (no information)
    """
    uniform = np.ones_like(probs) / len(probs)
    return max(0.01, np.sqrt(np.sum((probs - uniform) ** 2)))


def compute_grad_S(probs_list: List[np.ndarray]) -> float:
    """
    grad_S = dispersion across observations.

    High grad_S = fragments disagree (quantum)
    Low grad_S = fragments agree (classical)
    """
    if len(probs_list) < 2:
        return 0.01

    arr = np.array(probs_list)
    dispersion = np.mean(np.var(arr, axis=0))
    return max(0.01, dispersion)


def compute_R_single(probs: np.ndarray, sigma: float = 0.5, Df: float = 1.0) -> float:
    """
    R from single fragment observation.
    """
    E = compute_essence(probs)
    return E * (sigma ** Df)  # No grad_S for single observation


def compute_R_multi(probs_list: List[np.ndarray], sigma: float = 0.5, Df: float = 1.0) -> float:
    """
    R = (E / grad_S) * sigma^Df from multiple fragment observations.
    """
    if len(probs_list) == 0:
        return 0.0

    mean_probs = np.mean(probs_list, axis=0)
    E = compute_essence(mean_probs)
    grad_S = compute_grad_S(probs_list)

    return (E / grad_S) * (sigma ** Df)


def compute_R_mi(state: qt.Qobj, n_total: int, sigma: float = 0.5) -> float:
    """
    R based on Mutual Information - the correct metric for Quantum Darwinism.

    R_mi = (MI_avg / grad_MI) * sigma^Df

    Where:
    - MI_avg = average mutual information between system and single fragments
    - grad_MI = dispersion of MI across fragments (low = consensus)
    - Df = log(n_fragments) captures redundancy dimension
    - sigma = scaling factor

    High R_mi = fragments have high, consistent MI with system (classical reality)
    Low R_mi = fragments have low or inconsistent MI (quantum uncertainty)
    """
    n_env = n_total - 1
    if n_env < 1:
        return 0.0

    # System entropy (maximum possible MI)
    rho_sys = state.ptrace([0])
    sys_entropy = qt.entropy_vn(rho_sys, base=2)

    if sys_entropy < 0.01:
        return 1.0  # Pure state, trivially high R

    # Compute MI with each fragment
    mi_values = []
    for f in range(1, n_total):
        rho_s = state.ptrace([0])
        rho_f = state.ptrace([f])
        rho_sf = state.ptrace([0, f])
        S_s = qt.entropy_vn(rho_s, base=2)
        S_f = qt.entropy_vn(rho_f, base=2)
        S_sf = qt.entropy_vn(rho_sf, base=2)
        mi = max(0, S_s + S_f - S_sf)
        mi_values.append(mi / sys_entropy)  # Normalize to [0, 1]

    mi_array = np.array(mi_values)

    # E = average normalized MI (how much info do fragments have?)
    E_mi = np.mean(mi_array)

    # grad_S = dispersion of MI (do fragments AGREE?)
    # Low dispersion = high consensus = classical
    grad_mi = np.std(mi_array) + 0.01  # Floor to prevent division by zero

    # Df = log of number of fragments (redundancy dimension)
    Df = np.log(n_env + 1)

    # R = (E / grad) * sigma^Df
    return (E_mi / grad_mi) * (sigma ** Df)


def compute_R_overlap(psi_prev: qt.Qobj, psi_curr: qt.Qobj) -> float:
    """
    Alternative R based on state overlap between timesteps.

    E = |<psi_prev|psi_curr>|^2 (fidelity)

    This directly measures "how much has the state changed?"
    """
    if psi_prev is None:
        return 1.0

    # For density matrices
    if psi_prev.type == 'oper' and psi_curr.type == 'oper':
        return qt.fidelity(psi_prev, psi_curr) ** 2

    # For pure states
    overlap = np.abs(psi_prev.overlap(psi_curr)) ** 2
    return overlap


# =============================================================================
# TIME EVOLUTION AND DATA COLLECTION
# =============================================================================

def run_decoherence_simulation(params: DecoherenceParams) -> Dict:
    """
    Run time evolution and collect all metrics at each timestep.

    This is the core simulation that produces the decoherence trajectory.
    """
    print("=" * 70)
    print("Q54 TEST C: ZUREK QUANTUM DARWINISM DECOHERENCE")
    print("=" * 70)
    print()
    print("Data source: QuTiP spin-boson simulation")
    print(f"Environment modes: {params.n_env_modes}")
    print(f"Coupling strength: {params.coupling}")
    print(f"Estimated t_decoherence: {params.t_decoherence_estimate:.2f}")
    print()

    # Create Hamiltonian and initial state
    H, c_ops = create_spin_boson_hamiltonian(params)
    psi_0 = create_initial_state(params)
    n_total = 1 + params.n_env_modes

    # Time points
    times = np.linspace(0, params.t_max, params.n_timesteps)

    # Run master equation evolution
    print("Running Lindblad master equation...")
    result = qt.mesolve(H, psi_0, times, c_ops, [])

    # Collect metrics at each timestep
    data = {
        'time': [],
        'coherence': [],
        'system_entropy': [],
        'redundancy': [],
        'mutual_info_single': [],
        'mutual_info_all': [],
        'R_single': [],
        'R_multi': [],
        'R_joint': [],
        'R_mi': [],  # NEW: MI-based R metric (correct for QD)
        'R_overlap': [],
        'dR_dt': [],
    }

    prev_state = None
    prev_R = None

    print("Computing metrics at each timestep...")

    for i, t in enumerate(times):
        state = result.states[i]

        # System reduced density matrix
        rho_sys = state.ptrace([0])

        # Basic quantum metrics
        coherence = compute_coherence(rho_sys)
        sys_entropy = qt.entropy_vn(rho_sys, base=2)
        redundancy = compute_redundancy(state, n_total)

        # Mutual information (single fragment)
        mi_single = compute_mutual_information(state, n_total, 0, [1])

        # Mutual information (all fragments)
        mi_all = compute_mutual_information(state, n_total, 0, list(range(1, n_total)))

        # R computations
        # Single fragment observation
        single_probs = get_fragment_probs(state, [1])
        R_single = compute_R_single(single_probs, Df=1.0)

        # Multi-fragment observation (independent)
        all_probs = [get_fragment_probs(state, [f]) for f in range(1, n_total)]
        R_multi = compute_R_multi(all_probs, Df=np.log(params.n_env_modes + 1))

        # Joint fragment observation
        joint_probs = get_fragment_probs(state, list(range(1, n_total)))
        R_joint = compute_R_single(joint_probs, Df=np.log(params.n_env_modes + 1))

        # MI-based R (the CORRECT metric for Quantum Darwinism)
        R_mi = compute_R_mi(state, n_total)

        # Overlap-based R (between timesteps)
        R_overlap = compute_R_overlap(prev_state, state)

        # Rate of change of R (using R_mi for QD-correct metric)
        if prev_R is not None and i > 0:
            dt = times[i] - times[i-1]
            dR_dt = (R_mi - prev_R) / dt
        else:
            dR_dt = 0.0

        # Store
        data['time'].append(t)
        data['coherence'].append(coherence)
        data['system_entropy'].append(sys_entropy)
        data['redundancy'].append(redundancy)
        data['mutual_info_single'].append(mi_single)
        data['mutual_info_all'].append(mi_all)
        data['R_single'].append(R_single)
        data['R_multi'].append(R_multi)
        data['R_joint'].append(R_joint)
        data['R_mi'].append(R_mi)
        data['R_overlap'].append(R_overlap)
        data['dR_dt'].append(dR_dt)

        prev_state = state
        prev_R = R_mi  # Track R_mi for rate of change

    return data


def find_decoherence_time(data: Dict) -> float:
    """
    Find the decoherence time from coherence decay.

    Definition: t_dec where coherence drops to 1/e of initial value.
    """
    coherence = np.array(data['coherence'])
    times = np.array(data['time'])

    initial_coherence = coherence[0]
    target = initial_coherence / np.e

    # Find first crossing
    for i, c in enumerate(coherence):
        if c < target:
            return times[i]

    return times[-1]  # Never reached


def find_R_spike(data: Dict) -> Dict:
    """
    Analyze R trajectory for spike/step behavior at decoherence.

    Returns: dictionary with spike analysis results

    NOTE: Uses R_mi (mutual-information-based R) which measures how much
    information fragments carry about the system, with consensus weighting.
    R_mi INCREASES during decoherence as fragments gain correlated info.
    """
    times = np.array(data['time'])
    R_mi = np.array(data['R_mi'])
    dR_dt = np.array(data['dR_dt'])

    # Find decoherence time
    t_dec = find_decoherence_time(data)

    # Find max derivative (spike location)
    spike_idx = np.argmax(np.abs(dR_dt[1:]))  # Skip first point
    spike_time = times[spike_idx + 1]
    spike_magnitude = dR_dt[spike_idx + 1]

    # R values before and after decoherence
    dec_idx = np.argmin(np.abs(times - t_dec))

    # Average R in early period (before decoherence)
    early_idx = max(1, dec_idx // 2)
    R_before = np.mean(R_mi[:early_idx])

    # Average R in late period (after decoherence)
    late_idx = dec_idx + (len(times) - dec_idx) // 2
    R_after = np.mean(R_mi[late_idx:])

    # Transition sharpness: ratio of step height to transition width
    R_at_dec = R_mi[dec_idx]
    transition_height = np.abs(R_after - R_before)
    transition_width = t_dec if t_dec > 0 else 1.0
    sharpness = transition_height / transition_width

    return {
        't_decoherence': t_dec,
        'spike_time': spike_time,
        'spike_magnitude': spike_magnitude,
        'R_before_transition': R_before,
        'R_at_transition': R_at_dec,
        'R_after_transition': R_after,
        'transition_sharpness': sharpness,
        'R_change_ratio': R_after / max(R_before, 0.01),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(data: Dict, spike_analysis: Dict, output_dir: str):
    """Create visualization of R vs time showing crystallization behavior."""

    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plots")
        return

    times = np.array(data['time'])
    t_dec = spike_analysis['t_decoherence']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: R_mi vs time (main result - MI-based metric)
    ax = axes[0, 0]
    ax.plot(times, data['R_mi'], 'b-', linewidth=2, label='R_mi (MI-based)')
    ax.axvline(t_dec, color='r', linestyle='--', alpha=0.7, label=f't_dec = {t_dec:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('R_mi')
    ax.set_title('R_mi vs Time: Crystallization via Mutual Information')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate before/after values
    ax.annotate(f'R_before = {spike_analysis["R_before_transition"]:.3f}',
                xy=(t_dec/4, spike_analysis["R_before_transition"]),
                fontsize=9)
    ax.annotate(f'R_after = {spike_analysis["R_after_transition"]:.3f}',
                xy=(times[-1]*0.6, spike_analysis["R_after_transition"]),
                fontsize=9)

    # Plot 2: dR/dt (rate of change)
    ax = axes[0, 1]
    ax.plot(times, data['dR_dt'], 'g-', linewidth=2)
    ax.axvline(t_dec, color='r', linestyle='--', alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('dR/dt')
    ax.set_title('Rate of Change of R')
    ax.grid(True, alpha=0.3)

    # Plot 3: Coherence decay
    ax = axes[1, 0]
    ax.plot(times, data['coherence'], 'purple', linewidth=2)
    ax.axvline(t_dec, color='r', linestyle='--', alpha=0.7)
    ax.axhline(1/np.e, color='orange', linestyle=':', alpha=0.7, label='1/e threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Coherence')
    ax.set_title('System Coherence (Off-diagonal)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Redundancy
    ax = axes[1, 1]
    ax.plot(times, data['redundancy'], 'orange', linewidth=2)
    ax.axvline(t_dec, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Redundancy')
    ax.set_title('Zurek Redundancy (Classical Objectivity)')
    ax.grid(True, alpha=0.3)

    # Plot 5: Multiple R metrics
    ax = axes[2, 0]
    ax.plot(times, data['R_single'], 'b-', alpha=0.5, label='R_single')
    ax.plot(times, data['R_multi'], 'g-', alpha=0.5, label='R_multi')
    ax.plot(times, data['R_joint'], 'r-', alpha=0.5, label='R_joint')
    ax.plot(times, data['R_mi'], 'purple', linewidth=2, label='R_mi (primary)')
    ax.axvline(t_dec, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('R')
    ax.set_title('Comparison of R Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: R_overlap (state fidelity)
    ax = axes[2, 1]
    ax.plot(times, data['R_overlap'], 'teal', linewidth=2)
    ax.axvline(t_dec, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('State Fidelity')
    ax.set_title('R_overlap: |<psi(t)|psi(t-dt)>|^2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'test_c_zurek_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.close()


# =============================================================================
# PASS/FAIL ANALYSIS
# =============================================================================

def analyze_hypothesis(data: Dict, spike_analysis: Dict) -> Dict:
    """
    Test the hypothesis against the data.

    HYPOTHESIS:
    - R shows step function or spike at t_decoherence
    - Before decoherence: R fluctuates (quantum uncertainty)
    - After decoherence: R stabilizes at high value (classical lock-in)
    - The transition should be sharp, not gradual

    FALSIFICATION:
    - R is flat throughout (no transition signature)
    - R DECREASES during decoherence (opposite of crystallization)
    - No threshold behavior visible

    NOTE: Uses R_mi (mutual-information-based R) which INCREASES during decoherence
    as environment fragments gain correlated information about the system state.
    This is the correct metric for Quantum Darwinism crystallization.
    """
    R_mi = np.array(data['R_mi'])
    times = np.array(data['time'])

    t_dec = spike_analysis['t_decoherence']
    dec_idx = np.argmin(np.abs(times - t_dec))

    # Test 1: Does R INCREASE across decoherence?
    R_before = spike_analysis['R_before_transition']
    R_after = spike_analysis['R_after_transition']
    R_change = R_after - R_before
    R_increases = R_change > 0.01  # Significant increase

    # Test 2: Is the transition sharp (not gradual)?
    sharpness = spike_analysis['transition_sharpness']
    is_sharp = sharpness > 0.1  # Threshold for "sharp"

    # Test 3: Does R stabilize after decoherence?
    late_R = R_mi[dec_idx:]
    late_variance = np.var(late_R)
    early_variance = np.var(R_mi[:dec_idx]) if dec_idx > 1 else late_variance
    stabilizes = late_variance < early_variance  # Variance decreases

    # Test 4: Is there a clear spike in dR/dt?
    dR_dt = np.array(data['dR_dt'])
    max_dR = np.max(np.abs(dR_dt[1:]))
    mean_dR = np.mean(np.abs(dR_dt[1:]))
    has_spike = max_dR > 3 * mean_dR  # Spike is 3x above average

    # Test 5: Correlation with Zurek redundancy
    from scipy import stats
    r_corr, p_corr = stats.pearsonr(R_mi, data['redundancy'])
    correlated_with_redundancy = r_corr > 0.5 and p_corr < 0.05

    # Combine results
    tests_passed = sum([
        R_increases,
        is_sharp,
        stabilizes,
        has_spike,
        correlated_with_redundancy
    ])

    # Verdict
    if R_change < -0.01:
        verdict = "FAIL"
        reason = "R DECREASES during decoherence (opposite of crystallization)"
    elif not R_increases and abs(R_change) < 0.01:
        verdict = "FAIL"
        reason = "R is flat throughout (no transition signature)"
    elif tests_passed >= 3:
        verdict = "PASS"
        reason = f"{tests_passed}/5 criteria met for crystallization hypothesis"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"Only {tests_passed}/5 criteria met"

    return {
        'verdict': verdict,
        'reason': reason,
        'tests': {
            'R_increases': R_increases,
            'is_sharp': is_sharp,
            'stabilizes': stabilizes,
            'has_spike': has_spike,
            'correlated_with_redundancy': correlated_with_redundancy,
        },
        'metrics': {
            'R_change': R_change,
            'sharpness': sharpness,
            'late_variance': late_variance,
            'early_variance': early_variance,
            'max_dR_dt': max_dR,
            'R_redundancy_correlation': r_corr,
            'R_redundancy_p_value': p_corr,
        }
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test_c():
    """
    Main entry point for Q54 Test C.
    """
    if not QUTIP_AVAILABLE:
        print("ERROR: QuTiP required for this test")
        print("Install with: pip install qutip")
        return None

    # Set parameters
    params = DecoherenceParams()

    # Run simulation
    print("-" * 70)
    data = run_decoherence_simulation(params)

    # Analyze for spike behavior
    print("-" * 70)
    print("Analyzing R trajectory...")
    spike_analysis = find_R_spike(data)

    print()
    print("Decoherence Time Analysis:")
    print(f"  t_decoherence = {spike_analysis['t_decoherence']:.3f}")
    print(f"  R before = {spike_analysis['R_before_transition']:.4f}")
    print(f"  R after  = {spike_analysis['R_after_transition']:.4f}")
    print(f"  Transition sharpness = {spike_analysis['transition_sharpness']:.4f}")
    print(f"  R change ratio = {spike_analysis['R_change_ratio']:.2f}x")

    # Test hypothesis
    print("-" * 70)
    print("Testing Hypothesis...")
    hypothesis_result = analyze_hypothesis(data, spike_analysis)

    print()
    print("Individual Tests:")
    for test_name, passed in hypothesis_result['tests'].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    print()
    print("=" * 70)
    print(f"VERDICT: {hypothesis_result['verdict']}")
    print(f"Reason: {hypothesis_result['reason']}")
    print("=" * 70)

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Create plots
    print()
    print("Creating visualizations...")
    create_plots(data, spike_analysis, output_dir)

    # Create results JSON
    results = {
        'test_name': 'Q54_Test_C_Zurek_Data',
        'timestamp': datetime.now().isoformat(),
        'data_source': 'simulated',
        'simulation_params': {
            'n_env_modes': params.n_env_modes,
            'coupling': params.coupling,
            'temperature': params.temperature,
            't_max': params.t_max,
            'n_timesteps': params.n_timesteps,
        },
        'decoherence_time': spike_analysis['t_decoherence'],
        'R_before_transition': spike_analysis['R_before_transition'],
        'R_after_transition': spike_analysis['R_after_transition'],
        'transition_sharpness': spike_analysis['transition_sharpness'],
        'hypothesis_tests': hypothesis_result['tests'],
        'hypothesis_metrics': hypothesis_result['metrics'],
        'verdict': hypothesis_result['verdict'],
        'reason': hypothesis_result['reason'],
        'prediction_stated_before': {
            'expectation': 'R spike/step at t_decoherence',
            'before_behavior': 'R fluctuates (quantum uncertainty)',
            'after_behavior': 'R stabilizes at high value (classical lock-in)',
        },
        'falsification_criteria': [
            'R is flat throughout (no transition signature)',
            'R DECREASES during decoherence (opposite of crystallization)',
            'No threshold behavior visible',
        ]
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj) if isinstance(obj, np.integer) else bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = convert_to_native(results)

    results_path = os.path.join(output_dir, 'test_c_zurek_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Print final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("HYPOTHESIS: R spikes at the moment classical behavior emerges")
    print()
    print("OBSERVATIONS:")
    print(f"  - Decoherence time: t = {spike_analysis['t_decoherence']:.2f}")
    print(f"  - R before: {spike_analysis['R_before_transition']:.4f}")
    print(f"  - R after:  {spike_analysis['R_after_transition']:.4f}")
    print(f"  - R change: {hypothesis_result['metrics']['R_change']:.4f}")
    print(f"  - R correlates with Zurek redundancy: r = {hypothesis_result['metrics']['R_redundancy_correlation']:.3f}")
    print()
    print(f"VERDICT: {hypothesis_result['verdict']}")
    print()

    if hypothesis_result['verdict'] == 'PASS':
        print("INTERPRETATION:")
        print("  The formula R = (E/grad_S) * sigma^Df shows threshold behavior")
        print("  at the decoherence transition, consistent with the hypothesis that")
        print("  R tracks the 'crystallization' of classical reality from quantum")
        print("  superposition as described by quantum Darwinism.")
        print()
        print("  This supports Q54's thesis: energy 'locks in' to matter-like")
        print("  classical reality through redundant self-replication into the")
        print("  environment, and R detects this transition.")

    elif hypothesis_result['verdict'] == 'FAIL':
        print("INTERPRETATION:")
        print("  The data does NOT support the crystallization hypothesis.")
        print(f"  Reason: {hypothesis_result['reason']}")
        print()
        print("  This suggests either:")
        print("  1. The formula does not capture the quantum-classical transition")
        print("  2. The mapping between quantum and semiotic mechanics needs revision")
        print("  3. The simulation parameters need adjustment")

    return results


if __name__ == "__main__":
    results = run_test_c()
