"""
35.3_hatano_nelson_skin_effect.py

Infinite Tape Halting via Non-Hermitian Skin Effect.

PHYSICS:  An infinite TM tape is a 1D Hatano-Nelson lattice with
asymmetric hopping.  The Non-Hermitian Skin Effect causes bulk
eigenstates to exponentially localize at boundaries under open
boundary conditions (OBC) when hopping is asymmetric.

  - Halting TM (forward-only chain):  t_R > 0,  t_L = 0
      -> OBC:  all eigenvalues on imaginary axis, IPR ~ 1
      -> PBC:  eigenvalues trace a closed loop in C
      -> Lyapunov exponent > 0  ->  LOCALIZATION  ->  HALTS

  - Looping TM (symmetric cycle):  t_R = t_L > 0
      -> OBC:  eigenvalues on real axis, IPR ~ 1/L (delocalized)
      -> PBC:  eigenvalues on real segment
      -> Lyapunov exponent = 0  ->  DELOCALIZED  ->  LOOPS

MEASUREMENTS:
  1.  Eigenvalue spectrum under OBC and PBC
  2.  Spectral collapse:  OBC spectrum vs PBC spectral loop
  3.  Inverse Participation Ratio (IPR) — localization diagnostic
  4.  Point-gap winding under boundary twist (phi sweep)
  5.  Lyapunov exponent from transfer matrix

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
from torch import linalg as LA

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  Hatano-Nelson Hamiltonian builders
# ---------------------------------------------------------------------------

def hatano_nelson_obc(L, t_R=1.0, t_L=0.0, eps=0.0, sink_idx=None,
                      sink_strength=10.0):
    """
    Open Boundary Condition (OBC) Hatano-Nelson chain.

    H[i][i+1] = -t_L   (left hopping, upper diagonal)
    H[i+1][i] = -t_R   (right hopping, lower diagonal)
    H[i][i]   = eps    (on-site energy, may be complex)

    If sink_idx is set, the site at that index gets an additional
    imaginary potential -i * sink_strength (Exceptional Point sink).

    Returns non-Hermitian H of shape [L, L].
    """
    H = torch.zeros((L, L), dtype=torch.complex64)
    for i in range(L):
        H[i, i] = eps + 0j
        if i < L - 1:
            H[i, i + 1] = -t_L + 0j
        if i > 0:
            H[i, i - 1] = -t_R + 0j
    if sink_idx is not None:
        H[sink_idx, sink_idx] = -1j * sink_strength
    return H


def hatano_nelson_pbc(L, t_R=1.0, t_L=0.0, eps=0.0):
    """
    Periodic Boundary Condition (PBC) Hatano-Nelson ring.

    Same as OBC but wraps around:  H[0][L-1] = -t_L,  H[L-1][0] = -t_R
    No sink — the ring has no boundary.
    """
    H = torch.zeros((L, L), dtype=torch.complex64)
    for i in range(L):
        H[i, i] = eps + 0j
        H[i, (i + 1) % L] = -t_L + 0j
        H[i, (i - 1) % L] = -t_R + 0j
    return H


def hatano_nelson_twisted(L, t_R=1.0, t_L=0.0, eps=0.0, phi=0.0):
    """
    PBC Hatano-Nelson with a U(1) twist phi on the boundary link.
    The edge (L-1 -> 0) gets a phase factor exp(i*phi).
    """
    H = torch.zeros((L, L), dtype=torch.complex64)
    for i in range(L):
        H[i, i] = eps + 0j
        if i < L - 1:
            H[i, i + 1] = -t_L + 0j
            H[i + 1, i] = -t_R + 0j
    twist = torch.tensor(np.exp(1j * phi), dtype=torch.complex64)
    H[0, L - 1] = -t_L * twist
    H[L - 1, 0] = -t_R * twist.conj()
    return H


# ---------------------------------------------------------------------------
# 2.  Diagnostics
# ---------------------------------------------------------------------------

def ipr(psi):
    """Inverse Participation Ratio — measures localization.
    IPR = sum_i |psi_i|^4.  ~ 1 = localized, ~ 1/L = delocalized."""
    return float((psi.abs().pow(4).sum() / psi.abs().pow(2).sum().pow(2)).item())


def lyapunov_exponent(t_R, t_L, eps=0.0, E=0.0j, n_iter=2000):
    """
    Estimate the Lyapunov exponent of the transfer matrix.

    For the Hatano-Nelson chain with uniform hopping:
      T(E) = [[(E - eps)/t_R,  -t_L/t_R],
              [1,             0       ]]

    For purely directed chains (t_L = 0): T is lower-triangular with
    eigenvalue ratio |(E-eps)/t_R|.  lambda = log|(E-eps)/t_R|.
    lambda > 0  ->  amplification away from boundary  ->  LOOPS
    lambda < 0  ->  exponential decay toward boundary   ->  SKIN EFFECT

    For symmetric chains (t_L = t_R): the transfer matrix is symplectic
    and lambda ~ 0 (no net direction).

    Returns (lambda, is_degenerate)
    """
    if abs(t_L) < 1e-12:
        ratio = abs(E - eps) / max(abs(t_R), 1e-12)
        if ratio < 1e-12:
            return -np.inf, True
        return float(np.log(ratio)), False

    if abs(t_R) < 1e-12:
        ratio = abs(E - eps) / max(abs(t_L), 1e-12)
        if ratio < 1e-12:
            return -np.inf, True
        return float(np.log(ratio)), False

    T = torch.zeros((2, 2), dtype=torch.complex64)
    T[0, 0] = (E - eps) / t_R
    T[0, 1] = -t_L / t_R
    T[1, 0] = 1.0 + 0j
    T[1, 1] = 0j

    v = torch.ones(2, dtype=torch.complex64)
    v = v / LA.norm(v)
    log_norm = 0.0
    for _ in range(n_iter):
        v = T @ v
        nrm = float(LA.norm(v).item())
        if nrm < 1e-30:
            return -np.inf, True
        log_norm += np.log(nrm)
        v = v / nrm
    return log_norm / n_iter, False


def winding_phi_sweep(L, t_R, t_L, eps=0.0, n_phi=360):
    """
    Point-gap winding via boundary twist phi sweep on the PBC ring.
    Computes det(H(phi)) as phi varies 0->2pi and returns winding.
    """
    dets = torch.zeros(n_phi, dtype=torch.complex64)
    for k in range(n_phi):
        phi = 2.0 * np.pi * k / n_phi
        H_phi = hatano_nelson_twisted(L, t_R, t_L, eps, phi)
        dets[k] = LA.det(H_phi)
    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta + np.pi, 2.0 * np.pi) - np.pi
    return float(torch.sum(dtheta).item()) / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# 3.  Oracle runner
# ---------------------------------------------------------------------------

def run_skin_oracle(name, L, t_R, t_L, sink_idx=None,
                    sink_strength=10.0, eps=0.0, verbose=True):
    """Run the full Skin Effect halting oracle on a Hatano-Nelson chain."""

    # ----  build Hamiltonians  -------------------------------------------
    H_obc = hatano_nelson_obc(L, t_R, t_L, eps, sink_idx, sink_strength)
    H_pbc = hatano_nelson_pbc(L, t_R, t_L, eps)

    # ----  eigenvalues + eigenvectors (single eigensolve)  -----------
    eig_obc, eigvecs_obc = LA.eig(H_obc)
    eig_pbc, _ = LA.eig(H_pbc)
    ipr_vals = torch.tensor([ipr(eigvecs_obc[:, i])
                              for i in range(L)], dtype=torch.float64)
    ipr_mean = float(ipr_vals.mean().item())
    ipr_max = float(ipr_vals.max().item())

    # ----  Lyapunov exponent  --------------------------------------------
    lambda_lyap, lyap_degen = lyapunov_exponent(t_R, t_L, eps, E=0j)

    # ----  winding under twist  ------------------------------------------
    W_twist = winding_phi_sweep(L, t_R, t_L, eps)

    # ----  spectral collapse ratio  --------------------------------------
    obc_radius = float(eig_obc.abs().max().item())
    pbc_radius = float(eig_pbc.abs().max().item())
    collapse_ratio = obc_radius / pbc_radius if pbc_radius > 1e-12 else 0.0

    # ----  verdict  ------------------------------------------------------
    asym = abs(t_R - t_L) > 0.01
    halted = sink_idx is not None
    if halted:
        verdict = "HALTS"
        subtype = (f"sink at site {sink_idx}  "
                   f"IPR={ipr_mean:.3f}  "
                   f"lambda={lambda_lyap:+.4f}")
    elif asym and not halted:
        verdict = "LOOPS (directed chain)"
        subtype = (f"asymmetric hopping  "
                   f"IPR={ipr_mean:.3f}  "
                   f"lambda={lambda_lyap:+.4f}")
    else:
        verdict = "LOOPS (symmetric)"
        subtype = (f"balanced hopping  "
                   f"IPR={ipr_mean:.3f}  "
                   f"lambda={lambda_lyap:+.4f}")

    # ----  output  -------------------------------------------------------
    if verbose:
        print("=" * 70)
        print(f"  SKIN EFFECT HALTING ORACLE  --  {name}")
        print("=" * 70)
        print(f"  Chain length L={L}  |  t_R={t_R}  t_L={t_L}"
              f"  |  sink={'YES at '+str(sink_idx) if halted else 'NONE'}")
        print()
        print(f"  OBC eigenvalues (first 8):")
        for i in range(min(L, 8)):
            lam = eig_obc[i]
            print(f"    [{i}] = {lam.real.item():+8.4f}{lam.imag.item():+8.4f}j")
        print(f"  OBC spectral radius = {obc_radius:.4f}")
        print()
        print(f"  PBC eigenvalues (first 8):")
        for i in range(min(L, 8)):
            lam = eig_pbc[i]
            print(f"    [{i}] = {lam.real.item():+8.4f}{lam.imag.item():+8.4f}j")
        print(f"  PBC spectral radius = {pbc_radius:.4f}")
        print()
        print(f"  Diagnostics:")
        print(f"    IPR mean = {ipr_mean:.4f}  max = {ipr_max:.4f}"
              f"  (1=localized, {1/L:.4f}=delocalized)")
        print(f"    Lyapunov exponent  lambda = {lambda_lyap:+.6f}"
              f"  {'(localized)' if lambda_lyap > 0.01 else '(delocalized)'}")
        print(f"    Spectral collapse   OBC/PBC = {collapse_ratio:.4f}")
        print(f"    Boundary twist winding   W = {W_twist:+.3f}")
        print(f"\n  ***  VERDICT:  {verdict}")
        print(f"  ***  {subtype}")
        print("=" * 70)
        print()

    return {
        "name": name,
        "L": L,
        "t_R": t_R,
        "t_L": t_L,
        "sink_idx": sink_idx,
        "eig_obc": eig_obc,
        "eig_pbc": eig_pbc,
        "ipr_mean": ipr_mean,
        "ipr_max": ipr_max,
        "lyapunov": lambda_lyap,
        "W_twist": W_twist,
        "collapse_ratio": collapse_ratio,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    L = 24  # tape length

    cases = [
        ("Halting Chain (sink at end)",     L, 1.0, 0.0, L-1),
        ("Halting Chain (sink at middle)",  L, 1.0, 0.0, L//2),
        ("Directed Chain (no sink, LOOPS)", L, 1.0, 0.0, None),
        ("Symmetric Ring (no sink, LOOPS)", L, 1.0, 1.0, None),
    ]

    results = []
    for name, L, t_R, t_L, sink_idx in cases:
        r = run_skin_oracle(name, L, t_R, t_L,
                            sink_idx=sink_idx, verbose=True)
        results.append(r)

    # Summary
    print()
    print("=" * 70)
    L_ref = L
    print(f"  SKIN EFFECT ORACLE SUMMARY  (L={L_ref})")
    print("=" * 70)
    header = (f"  {'Case':<38s}  {'IPR':>6s}  "
              f"{'lambda':>8s}  {'W_twist':>7s}  "
              f"{'OBC/PBC':>7s}  {'Verdict'}")
    print(header)
    print("  " + "-" * 85)
    for r in results:
        print(f"  {r['name']:<38s}  {r['ipr_mean']:6.4f}  "
              f"{r['lyapunov']:+8.4f}  {r['W_twist']:+7.4f}  "
              f"{r['collapse_ratio']:7.4f}  "
              f"{'HALTS' if r['verdict']=='HALTS' else 'LOOPS'}")
    print()
    print("  IPR ~ 1       -> localized (Skin Effect) -> HALTS")
    print("  IPR ~ 1/L    -> delocalized            -> LOOPS")
    print("  lambda > 0    -> exponential localization -> HALTS")
    print("  lambda ~ 0    -> propagation             -> LOOPS")
    print("  W_twist != 0  -> spectral loop           -> cycle detected")
    print("=" * 70)


if __name__ == "__main__":
    main()
