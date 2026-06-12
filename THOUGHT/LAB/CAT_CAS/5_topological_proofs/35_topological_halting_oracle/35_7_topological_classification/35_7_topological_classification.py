"""
35_7_topological_classification.py

Topological classification of the TM halting Hamiltonian under
non-Hermitian symmetry classes (38-fold way, Kawabata et al. 2019).

SYMMETRY ANALYSIS:
  The TM Hamiltonian has:
    - No time-reversal symmetry (complex entries, H != H^*)
    - No particle-hole symmetry
    - No chiral symmetry
    - No sublattice symmetry
  -> Class A (no symmetries).

TOPOLOGICAL INVARIANT (1D, Class A, point-gap):
  W = (1/2pi i) \oint dE log det(H - E_ref I)
  W is Z-valued.  W=0: trivial (halting).  W!=0: topological (looping).

PHASE DIAGRAM:
  Sweep coupling gamma and loss_rate to map the W=0 / W!=0 boundary.
  The EP transition line marks where the spectrum pinches.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
from torch import linalg as LA

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  Hamiltonian builder (non-Hermitian, from 35.2)
# ---------------------------------------------------------------------------

def build_nonhermitian_H(transitions, num_states, halt_idx=None,
                         gamma=1.0, loss_rate=0.1, halt_mult=10.0):
    symbols = 2
    N = num_states * symbols
    H = torch.zeros((N, N), dtype=torch.complex64)
    for s in range(num_states):
        for b in range(symbols):
            idx = s * symbols + b
            is_halt = (halt_idx is not None and s == halt_idx)
            H[idx, idx] = -1j * (halt_mult if is_halt else 1.0) * loss_rate
    for (s, b), (sn, bn, _dir) in transitions.items():
        i = s * symbols + b
        j = sn * symbols + bn
        H[j, i] = gamma + 0j
    return H


# ---------------------------------------------------------------------------
# 2.  Topological invariants
# ---------------------------------------------------------------------------

def point_gap_winding(H, n_phi=200):
    """Point-gap winding number: W = (1/2pi) sum Delta arg det(H(phi)).
    Twists ALL transitions globally.
    """
    N = H.shape[0]
    dets = torch.zeros(n_phi, dtype=torch.complex64)
    for k in range(n_phi):
        phi = 2.0 * np.pi * k / n_phi
        twist = torch.tensor(np.exp(1j * phi), dtype=torch.complex64)
        H_phi = H.clone()
        for i in range(N):
            for j in range(N):
                if i != j and H_phi[j, i].abs().item() > 1e-12:
                    H_phi[j, i] = H_phi[j, i] * twist
        dets[k] = LA.det(H_phi)
    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta + np.pi, 2.0 * np.pi) - np.pi
    return int(round(float(torch.sum(dtheta).item()) / (2.0 * np.pi)))


def ep_condition_number(H):
    """Eigenvector matrix condition number — diverges at EP."""
    _, eigvecs = LA.eig(H)
    return float(LA.cond(eigvecs).item())


def symmetry_check(H):
    """Determine symmetry class of H under AZ+ classification.
    Class A: no symmetries.  AI: TRS T^2=+1.  AII: TRS T^2=-1.
    D: PHS C^2=+1.  C: PHS C^2=-1.  AIII: chiral.
    BDI: TRS+PHS with T^2=C^2=+1."""
    # TRS with T=I: H = H* (real matrix)
    is_real = torch.allclose(H.imag, torch.zeros_like(H.imag), atol=1e-6)
    # PHS with C=I: H = -H* (anti-real)
    ph_sym = torch.allclose(H, -H.conj(), atol=1e-6)
    # Chiral with S=I: H = -H^T (anti-symmetric)
    chiral = torch.allclose(H, -H.T, atol=1e-6)

    if is_real and ph_sym:  return "BDI (TRS+PHS)"
    if is_real:              return "AI  (TRS, T^2=+1)"
    if ph_sym:               return "D   (PHS, C^2=+1)"
    if chiral:               return "AIII (chiral)"
    return "A   (no symmetries)"


# ---------------------------------------------------------------------------
# 3.  Phase diagram sweep
# ---------------------------------------------------------------------------

def sweep_phase_diagram():
    """Sweep (gamma, loss_rate) to map topological phase boundaries."""

    # Test on the Halt Chain (3-state) and Loop 3-Cycle
    transitions_halt = {(0, 0): (1, 0, 0), (1, 0): (2, 0, 0)}
    transitions_loop = {(0, 0): (1, 0, 0), (1, 0): (2, 0, 0),
                        (2, 0): (0, 0, 0)}

    gamma_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    loss_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    print("=" * 70)
    print("  TOPOLOGICAL PHASE DIAGRAM — W(gamma, loss_rate)")
    print("=" * 70)
    print(f"  {'gamma':>7s}  {'loss':>7s}  "
          f"{'W_halt':>7s}  {'W_loop':>7s}  "
          f"{'kappa_h':>10s}  {'Phase'}")
    print("  " + "-" * 55)

    phase_data = []
    for gamma in gamma_vals:
        for lr in loss_vals:
            H_h = build_nonhermitian_H(transitions_halt, 3, 2,
                                       gamma=gamma, loss_rate=lr)
            H_l = build_nonhermitian_H(transitions_loop, 3, None,
                                       gamma=gamma, loss_rate=lr)
            W_h = point_gap_winding(H_h)
            W_l = point_gap_winding(H_l)
            k_h = ep_condition_number(H_h)

            phase = "HALT=0 LOOP!=0" if W_h == 0 and W_l != 0 else \
                    "BOTH=0" if W_h == 0 and W_l == 0 else \
                    "BOTH!=0" if W_h != 0 and W_l != 0 else \
                    "HALT!=0 (ANOMALOUS)"
            phase_data.append((gamma, lr, W_h, W_l, k_h, phase))

        for gamma, lr, W_h, W_l, k_h, phase in phase_data[-len(loss_vals):]:
            ep_mark = "**EP**" if k_h > 1e6 else ""
            print(f"  {gamma:7.2f}  {lr:7.3f}  {W_h:7d}  {W_l:7d}  "
                  f"{k_h:>10.2e}  {phase} {ep_mark}")

    # Count phases
    halt_zero_loop_nonzero = sum(1 for _, _, W_h, W_l, _, _ in phase_data
                                 if W_h == 0 and W_l != 0)
    both_zero = sum(1 for _, _, W_h, W_l, _, _ in phase_data
                    if W_h == 0 and W_l == 0)
    both_nonzero = sum(1 for _, _, W_h, W_l, _, _ in phase_data
                       if W_h != 0 and W_l != 0)

    print(f"\n  Phase statistics:")
    print(f"    W_halt=0, W_loop!=0 (correct):     {halt_zero_loop_nonzero}")
    print(f"    W_halt=0, W_loop=0  (degenerate): {both_zero}")
    print(f"    W_halt!=0, W_loop!=0 (degenerate): {both_nonzero}")
    print("=" * 70)

    return phase_data


# ---------------------------------------------------------------------------
# 4.  Symmetry analysis
# ---------------------------------------------------------------------------

def symmetry_report():
    """Classify all 4 test machines under AZ+ symmetry classes."""

    machines = [
        ("Halt Direct",  {(0, 0): (1, 0, 0)}, 2, 1),
        ("Halt Chain",   {(0, 0): (1, 0, 0), (1, 0): (2, 0, 0)}, 3, 2),
        ("Loop 2-Cycle", {(0, 0): (1, 0, 0), (1, 0): (0, 0, 0)}, 2, None),
        ("Loop 3-Cycle", {(0, 0): (1, 0, 0), (1, 0): (2, 0, 0),
                          (2, 0): (0, 0, 0)}, 3, None),
    ]

    print("\n" + "=" * 70)
    print("  SYMMETRY CLASSIFICATION (38-fold way)")
    print("=" * 70)
    print(f"  {'Machine':<18s}  {'N':>3s}  {'Symmetry Class':<24s}  "
          f"{'W':>3s}  {'kappa(V)':>10s}  {'Topological?'}")
    print("  " + "-" * 75)

    for name, transitions, ns, hi in machines:
        H = build_nonhermitian_H(transitions, ns, hi)
        W = point_gap_winding(H)
        kappa = ep_condition_number(H)
        sym = symmetry_check(H)
        topo = "YES (Z invariant)" if W != 0 else "NO (trivial)"
        print(f"  {name:<18s}  {H.shape[0]:3d}  {sym:<24s}  "
              f"{W:3d}  {kappa:>10.2e}  {topo}")

    print()
    print("  Class A: no symmetries, Z topological invariant in 1D")
    print("  Topological = W != 0 = LOOPS = non-trivial point-gap winding")
    print("  Trivial      = W = 0  = HALTS = all eigenvalues collapsible")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main():
    # Symmetry classification
    symmetry_report()

    # Phase diagram
    print()
    sweep_phase_diagram()

    print("\n" + "=" * 70)
    print("  TOPOLOGICAL CLASSIFICATION SUMMARY")
    print("=" * 70)
    print("  Symmetry class:      Class A (no TRS, no PHS, no chiral)")
    print("  Spatial dimension:   1D (tape chain)")
    print("  Point-gap invariant: Z (integer winding number)")
    print("  W = 0:  trivial topology  ->  HALTS (Skin Effect)")
    print("  W != 0: topological phase  ->  LOOPS (spectral loop)")
    print("  Phase boundary:      EP formation (kappa(V) divergence)")
    print("=" * 70)


if __name__ == "__main__":
    main()
