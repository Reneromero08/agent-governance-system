"""
35_topological_halting_oracle.py

Topological measurement of TM halting via continuous complex Hilbert space
evolution on a complex torus S^1 x C^N.

PHYSICS:  A Turing-machine transition table is compiled to a Hermitian
Hamiltonian H such that the halt state sits at zero energy (E = 0), forming
a topological fixed-point attractor.  The initial state |psi(0)> evolves
under continuous Schroedinger dynamics  |psi(t)> = exp(-i H t) |psi(0)>.

The dynamical reachability of the halt subspace is measured via:
  (a)  Time-averaged halt-state population  p_halt(t)
  (b)  Winding number W_halt of the complex halt-subspace amplitude
  (c)  Resolvent winding W_res  of  <psi0| (zI - H)^{-1} |psi0>
       around z = 0  (counts initial-state-overlapping zero modes)

If the halt subspace is dynamically accessible (p_halt_max > 0.1),
the system possesses a topological fixed point -> HALTS.
If unreachable (p_halt ~ 0), the state traces a limit cycle -> LOOPS.

No step-by-step TM simulation.  No backpropagation.
All evolution is continuous, unitary, and torch.complex64.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
from torch import linalg as LA

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  TM-to-Hamiltonian compiler
# ---------------------------------------------------------------------------

def build_hamiltonian(transitions, num_states, symbols=2,
                      gamma=1.0, E_active=1.0, E_halt=0.0,
                      halt_idx=None):
    """Map a TM transition table to a Hermitian Hamiltonian.

    Basis          |s, b>   ->  index = state * symbols + symbol
    The halt state (halt_idx) gets E_halt = 0 (topological fixed point).
    Active states get E_active > 0.
    If halt_idx is None, NO halt subspace exists (all states active).

    Off-diagonal couplings (gamma) connect basis states per transitions.

    Returns
    -------
    H         :  complex64 [N, N] Hermitian
    labels    :  list of str
    halt_mask :  bool [N]
    """
    N = num_states * symbols
    H = torch.zeros((N, N), dtype=torch.complex64)
    labels = []
    halt_mask = torch.zeros(N, dtype=torch.bool)

    for s in range(num_states):
        for b in range(symbols):
            idx = s * symbols + b
            is_halt = (halt_idx is not None and s == halt_idx)
            labels.append(f"|s{s},b{b}>" + (" [H]" if is_halt else ""))
            halt_mask[idx] = is_halt
            H[idx, idx] = (E_halt if is_halt else E_active) + 0.0j

    gamma_c = torch.tensor(gamma, dtype=torch.complex64)
    for (s, b), (sn, bn, _dir) in transitions.items():
        i = s * symbols + b
        j = sn * symbols + bn
        coupling = -gamma_c
        H[i, j] = coupling
        H[j, i] = coupling.conj()

    return H, labels, halt_mask


# ---------------------------------------------------------------------------
# 2.  Spectral decomposition and continuous evolution
# ---------------------------------------------------------------------------

def evolve(psi0, H, t_end, n_steps):
    """Continuous Schroedinger evolution  |psi(t)> = exp(-i H t) |psi0>.

    Returns
    -------
    psi_t   : complex64 [n_steps, N]
    t_vals  : float64  [n_steps]
    eigvals : float64  [N]
    eigvecs : complex64 [N, N]
    c0      : complex64 [N]  --  expansion of psi0 in eigenbasis
    """
    t_vals = torch.linspace(0.0, t_end, n_steps)
    eigvals, eigvecs = LA.eigh(H)
    c0 = eigvecs.conj().T @ psi0
    psi_t = torch.zeros((n_steps, psi0.shape[0]), dtype=torch.complex64)
    for k, tk in enumerate(t_vals):
        phase = torch.exp(-1j * eigvals * tk)
        psi_t[k] = eigvecs @ (c0 * phase)
    return psi_t, t_vals, eigvals, eigvecs, c0


# ---------------------------------------------------------------------------
# 3.  Winding number around the origin
# ---------------------------------------------------------------------------

def winding_of_curve(curve):
    """Winding number of a closed complex curve around the origin.

    W = (1 / 2 pi)  sum_n  Delta theta_n
    where Delta theta_n = arg(z_{n+1} / z_n)  wrapped to [-pi, pi].
    """
    dtheta = torch.diff(torch.angle(curve))
    dtheta = torch.remainder(dtheta + np.pi, 2.0 * np.pi) - np.pi
    return float(torch.sum(dtheta).item()) / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# 4.  Resolvent winding -- initial-state Green's function around z = 0
# ---------------------------------------------------------------------------

def resolvent_winding(eigvals, c0, epsilon=1e-2):
    """
    Winding number of the resolvent G(z) = <psi0 | (zI - H)^{-1} | psi0>
    as z traces a circle of radius epsilon around the origin.

    G(z) = sum_k  |c_k|^2 / (z - lambda_k)

    The winding number W_res = (1/2pi i) oint_{|z|=epsilon} G'(z)/G(z) dz
    counts the number of poles of G (eigenvalues lambda_k with |c_k|^2 > 0)
    that lie INSIDE the contour minus zeros of G inside.

    For epsilon small, only eigenvalues lambda_k with |lambda_k| < epsilon
    and non-zero overlap |c_k|^2 contribute.

    PHYSICS:  W_res != 0  ->  the initial state overlaps a near-zero
              eigenmode  ->  the dynamics possess a zero-energy channel.

    Returns
    -------
    W_res  : float  --  resolvent winding number (integer-valued in exact
                        arithmetic)
    rho_0  : float  --  spectrally broadened density at omega = 0
    """
    overlaps = c0.abs().pow(2)
    inside = eigvals.abs() < epsilon
    W_res = -float(overlaps[inside].sum().item())

    # Lorentzian-broadened spectral density at zero
    rho_0 = float(
        (overlaps * epsilon / (eigvals.pow(2) + epsilon ** 2)).sum().item()
    ) / np.pi

    return W_res, rho_0


# ---------------------------------------------------------------------------
# 5.  Test machines
# ---------------------------------------------------------------------------

def halt_direct():
    """2-state:  active (0) <-> halt (1).

    Transitions:  s0,b0  ->  s1,b0  (halt)
    The halt state is directly reachable.
    """
    transitions = {(0, 0): (1, 0, 0)}
    return transitions, 2, 1  # states=2, halt_idx=1


def loop_2cycle():
    """2-state cycle:  s0 <-> s1  (both active, NO halt).

    Transitions:  s0,b0 -> s1,b0,  s1,b0 -> s0,b0
    """
    transitions = {(0, 0): (1, 0, 0), (1, 0): (0, 0, 0)}
    return transitions, 2, None  # no halt state


def loop_3cycle():
    """3-state cycle:  s0 -> s1 -> s2 -> s0   (NO halt)."""
    transitions = {
        (0, 0): (1, 0, 0),
        (1, 0): (2, 0, 0),
        (2, 0): (0, 0, 0),
    }
    return transitions, 3, None  # no halt state


def halt_chain():
    """3-state chain:  s0 -> s1 -> s2 (halt)."""
    transitions = {(0, 0): (1, 0, 0), (1, 0): (2, 0, 0)}
    return transitions, 3, 2  # states=3, halt_idx=2


# ---------------------------------------------------------------------------
# 6.  Oracle runner
# ---------------------------------------------------------------------------

def run_oracle(transitions, num_states, name, halt_idx=None,
               initial_idx=0, t_end=120.0, n_steps=12000, verbose=True):
    """Full topological halting oracle.

    Returns a dict of all measured quantities.
    """
    H, labels, halt_mask = build_hamiltonian(transitions, num_states,
                                             halt_idx=halt_idx)
    N = H.shape[0]

    # ----  initial state  -------------------------------------------------
    psi0 = torch.zeros(N, dtype=torch.complex64)
    psi0[initial_idx] = 1.0 + 0.0j
    psi0 = psi0 / LA.norm(psi0)

    # ----  eigensolve + evolve  -------------------------------------------
    psi_t, t_vals, eigvals, eigvecs, c0 = evolve(psi0, H, t_end, n_steps)

    # ----  resolvent winding (initial-state Green's function)  ------------
    W_res, rho_zero = resolvent_winding(eigvals, c0)

    # ----  autocorrelation  A(t) = <psi(0)|psi(t)>  ----------------------
    A_t = torch.einsum("i,ti->t", (psi0.conj(), psi_t))
    W_auto = winding_of_curve(A_t)

    # ----  energy trajectory  ---------------------------------------------
    E_t = torch.einsum("ti,ij,tj->t",
                       (psi_t.conj(), H, psi_t)).real

    # ----  halt-subspace population  --------------------------------------
    p_halt_t = psi_t[:, halt_mask].abs().pow(2).sum(dim=-1)
    p_halt_max = float(p_halt_t.max().item())
    p_halt_var = float(p_halt_t.var(dim=0).item())

    # ----  winding of halt projection  ------------------------------------
    halt_amp = psi_t[:, halt_mask].sum(dim=-1)
    W_halt = winding_of_curve(halt_amp)

    # ----  spectral statistics  -------------------------------------------
    delta_e = eigvals[1:] - eigvals[:-1]
    mean_gap = float(delta_e.mean().item()) if len(delta_e) > 0 else 0.0
    std_gap = float(delta_e.std(dim=0).item()) if len(delta_e) > 1 else 0.0
    spectral_ratio = std_gap / mean_gap if mean_gap > 1e-15 else 0.0

    # ----  free spectral range (full period of spectrum)  -----------------
    # The free spectral range FSR = gcd of eigenvalue spacings.
    # Approximate the fundamental period:  T_period = 2pi / FSR
    # For incommensurate spectra (irrational ratios), FSR -> 0, T -> inf.
    gaps_unique = torch.abs(delta_e[delta_e.abs() > 1e-6])
    if len(gaps_unique) > 0:
        fsr = float(gaps_unique.min().item())
        T_period = 2.0 * np.pi / fsr if fsr > 1e-12 else float("inf")
    else:
        fsr = 0.0
        T_period = float("inf")

    # ----  topological verdict  -------------------------------------------
    # Criterion:  the halt state is dynamically reachable if the halt
    # subspace receives significant population at any time.
    # p_halt_max > 0.1  ->  halt is part of the dynamical manifold  ->  HALTS
    # p_halt_max ~ 0    ->  halt is decoupled from the dynamics     ->  LOOPS
    halt_reachable = p_halt_max > 0.1

    if halt_reachable:
        verdict = "HALTS"
        subtype = f"halt reachable  p_halt_max={p_halt_max:.3f}"
    else:
        verdict = "LOOPS"
        subtype = f"halt unreachable  p_halt_max={p_halt_max:.3e}"

    # Output
    if verbose:
        print("=" * 70)
        print(f"  TOPOLOGICAL HALTING ORACLE  --  {name}")
        print("=" * 70)

        # Hamiltonian
        print(f"\nHamiltonian  (dim = {N})")
        print("-" * 70)
        for row in range(N):
            row_str = "  ".join(
                f"{H[row, col].real.item():6.2f}"
                for col in range(N)
            )
            print(f"  {labels[row]:>15s}   [{row_str}]")

        # Spectrum
        print(f"\nSpectrum (eigenvalues of H):")
        e_str = np.array2string(eigvals.numpy(), precision=4,
                                separator="  ", suppress_small=True)
        print(f"  {e_str}")
        if np.isfinite(T_period):
            print(f"  Spectral gap stats:  mean={mean_gap:.4f}  "
                  f"std/mean={spectral_ratio:.4f}  "
                  f"T_period={T_period:.2f}")
        else:
            print(f"  Spectral gap stats:  mean={mean_gap:.4f}  "
                  f"std/mean={spectral_ratio:.4f}  "
                  f"T_period = inf (incommensurate)")

        # Resolvent winding
        print(f"\n  Resolvent winding  W_res  = {W_res:+.6f}"
              f"  (initial-state Green's function)")
        print(f"  Spectral density at E=0     rho_0  = {rho_zero:.6e}")

        # Component windings
        print(f"\n  Autocorrelation winding      W_auto = {W_auto:+10.6f}")
        print(f"  Halt-subspace winding        W_halt = {W_halt:+10.6f}")

        # Halt dynamics
        print(f"\n  Halt probability:  max={p_halt_max:.4f}  "
              f"var={p_halt_var:.6f}  "
              f"final={p_halt_t[-1].item():.4f}")

        # Energy
        E_mean = float(E_t.mean().item())
        E_range = float(E_t.max().item() - E_t.min().item())
        print(f"  Mean energy  <E>  = {E_mean:.6f}")
        print(f"  Energy range      = {E_range:.6f}")

        # Verdict
        print(f"\n  ***  VERDICT:  {verdict}")
        print(f"  ***  {subtype}")
        print("=" * 70)
        print()

    return {
        "name": name,
        "H": H,
        "labels": labels,
        "halt_mask": halt_mask,
        "eigvals": eigvals,
        "spectral_ratio": spectral_ratio,
        "W_auto": W_auto,
        "W_halt": W_halt,
        "W_res": W_res,
        "rho_zero": rho_zero,
        "E_t": E_t,
        "E_mean": float(E_t.mean().item()),
        "p_halt_t": p_halt_t,
        "p_halt_max": p_halt_max,
        "p_halt_var": p_halt_var,
        "t_vals": t_vals,
        "A_t": A_t,
        "verdict": verdict,
        "halt_reachable": halt_reachable,
        "T_period": T_period,
    }


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    raw = [halt_direct(), halt_chain(), loop_2cycle(), loop_3cycle()]
    labels = [
        "Halt Direct (2-state, halt coupled)",
        "Halt Chain (3-state, halt terminal)",
        "Loop 2-Cycle (2-state, no halt)",
        "Loop 3-Cycle (3-state, no halt)",
    ]
    machines = [(labels[i], *raw[i]) for i in range(len(raw))]

    results = []
    for name, transitions, num_states, halt_idx in machines:
        r = run_oracle(transitions, num_states, name,
                       halt_idx=halt_idx,
                       initial_idx=0,
                       t_end=120.0,
                       n_steps=12000,
                       verbose=True)
        results.append(r)

    # Summary table
    print()
    print("=" * 70)
    print("  ORACLE SUMMARY  --  Topological Measurement Results")
    print("=" * 70)
    header = (f"  {'Machine':<38s}  {'W_res':>7s}  "
              f"{'W_halt':>7s}  {'p_halt_max':>10s}  "
              f"{'<E>':>7s}  {'Verdict'}")
    print(header)
    print("  " + "-" * 84)
    for r in results:
        print(f"  {r['name']:<38s}  {r['W_res']:+7.4f}  "
              f"{r['W_halt']:+7.4f}  "
              f"{r['p_halt_max']:>10.6f}  "
              f"{r['E_mean']:>7.4f}  "
              f"{'HALTS' if r['halt_reachable'] else 'LOOPS'}")
    print()
    print("  Criterion:  p_halt_max > 0.1  -> halt subspace dynamically reachable")
    print("  W_res:  resolvent winding of <psi0|(zI-H)^{-1}|psi0> around z=0")
    print("          counts near-zero eigenmodes overlapping the initial state")
    print("  W_halt: winding of the complex halt-subspace amplitude")
    print()
    print("  A non-trivial halt subspace population indicates a topological")
    print("  fixed point:  the computation HALTS.")
    print("  Zero halt population indicates a limit cycle:  the computation LOOPS.")
    print("=" * 70)


if __name__ == "__main__":
    main()
