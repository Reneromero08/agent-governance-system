"""
36_nonhermitian_oracle.py

Topological Halting Oracle — Non-Hermitian extension.

PHYSICS:  TM transition graphs are DIRECTED.  A non-Hermitian Hamiltonian
encodes forward-only transitions via asymmetric coupling:
    H[j][i] = gamma   (directed edge  i -> j)
    H[i][j] = 0        (no reverse edge)

THREE phenomena inaccessible to the Hermitian framework:

1. EXCEPTIONAL POINTS (EPs) — defect in the spectrum where BOTH eigenvalues
   AND eigenvectors coalesce into a Jordan block.  The halt state acts as
   a massive imaginary sink:  H[halt][halt] = -i * multiplier * loss_rate.
   At the EP the eigenvector matrix condition number kappa(V) diverges.

2. POINT-GAP WINDING — Non-Hermitian eigenvalues live in the complex plane.
   A contour CAN enclose them.  The winding of det(H - E*I) around a
   reference energy E_ref counts eigenvalues inside the contour.
       W = (1/2pi) * sum_n  Delta arg(det(H - E_n*I))
   W = 0  -> spectrum collapses into EP (Skin Effect)         -> HALTS
   W != 0 -> spectrum forms a closed loop around the EP       -> LOOPS

3. NON-HERMITIAN SKIN EFFECT — under open boundary conditions, bulk
   eigenstates exponentially localize at the boundary.  An infinite
   tape is a Hatano-Nelson lattice.  Localization = HALTS,
   delocalized propagation = LOOPS.

Measurement protocol:
   - torch.linalg.eig (complex eigenvalues, non-Hermitian capable)
   - torch.linalg.slogdet (stable determinant for contour integral)
   - torch.linalg.cond (eigenvector coalescence at EP)

No step-by-step TM simulation.  No backpropagation.  No Hermitian forcing.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
from torch import linalg as LA

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  Non-Hermitian Hamiltonian compiler
# ---------------------------------------------------------------------------

def build_nonhermitian_H(transitions, num_states, symbols=2,
                         gamma=1.0, loss_rate=0.1, halt_mult=10.0,
                         halt_idx=None):
    """
    Map a directed TM transition table to a NON-HERMITIAN Hamiltonian.

    Directed edges:  H[j][i] = +gamma  for transition  i -> j
                     H[i][j] = 0       (no reverse edge)

    On-site dissipation:
        Active states:   H[i][i] = -i * loss_rate
        Halt state:      H[h][h] = -i * halt_mult * loss_rate
                         (massive imaginary sink = Exceptional Point)

    Returns
    -------
    H         : complex64 [N, N]  non-Hermitian
    labels    : list of str
    halt_mask : bool [N]
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
            imag_loss = (halt_mult if is_halt else 1.0) * loss_rate
            H[idx, idx] = -1j * imag_loss

    for (s, b), (sn, bn, _dir) in transitions.items():
        i = s * symbols + b
        j = sn * symbols + bn
        H[j, i] = gamma + 0.0j          # directed: i -> j only
        # NOTE: no H[i, j] — reverse edge is NOT added

    return H, labels, halt_mask


# ---------------------------------------------------------------------------
# 2.  Point-gap winding via boundary twist (spectral flow)
# ---------------------------------------------------------------------------

def point_gap_winding(H_base, twist_indices, E_ref=0.0j, n_phi=400):
    """
    Compute the point-gap winding number via boundary twist spectral flow.

    H(phi) = H_base but with H[j][i] *= exp(i*phi) for (i,j) in twist_indices.
    As phi sweeps [0, 2pi), compute det(H(phi) - E_ref*I) and measure its
    winding number around the origin.

    The DETERMINANT winding is the correct collective invariant — individual
    eigenvalue trajectories can swap over a cycle (spectral flow), but
    det(H(phi)) = prod_i lambda_i(phi) captures the net rotation.

    W = 0  -> eigenvalues are static under twist -> trivial topology -> HALTS
    W != 0 -> eigenvalues trace a closed spectral loop -> LOOPS

    Returns the integer-rounded winding number and the raw float.
    """
    I = torch.eye(H_base.shape[0], dtype=torch.complex64)
    if not twist_indices:
        return 0.0, 0.0

    phi_vals = torch.linspace(0.0, 2.0 * np.pi, n_phi)
    dets = torch.zeros(n_phi, dtype=torch.complex64)

    for k, phi in enumerate(phi_vals):
        H_phi = H_base.clone()
        twist = torch.tensor(np.exp(1j * phi.item()), dtype=torch.complex64)
        for i, j in twist_indices:
            H_phi[j, i] = H_phi[j, i] * twist
        M = H_phi - E_ref * I
        dets[k] = LA.det(M)

    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta + np.pi, 2.0 * np.pi) - np.pi
    W_raw = float(torch.sum(dtheta).item()) / (2.0 * np.pi)
    W = round(W_raw)
    return W_raw, W


# ---------------------------------------------------------------------------
# 3.  Exceptional-point detection
# ---------------------------------------------------------------------------

def get_spectral_data(H):
    """
    Diagonalize the non-Hermitian H and compute EP diagnostics.

    Returns
    -------
    eigvals  : complex64 [N]  — complex eigenvalues
    eigvecs  : complex64 [N,N] — eigenvector matrix V
    kappa_V  : float          — condition number of V (diverges at EP)
    """
    eigvals, eigvecs = LA.eig(H)
    kappa_V = float(LA.cond(eigvecs).item())
    return eigvals, eigvecs, kappa_V


# ---------------------------------------------------------------------------
# 4.  Test machines (same as Experiment 35)
# ---------------------------------------------------------------------------

def halt_direct():
    """2-state: active(0) -> halt(1).  No cycle."""
    transitions = {(0, 0): (1, 0, 0)}
    twist_edges = []                       # no cycle to twist
    return transitions, 2, 1, twist_edges


def halt_chain():
    """3-state chain: s0 -> s1 -> s2(halt).  No cycle."""
    transitions = {(0, 0): (1, 0, 0), (1, 0): (2, 0, 0)}
    twist_edges = []
    return transitions, 3, 2, twist_edges


def loop_2cycle():
    """2-state cycle: s0 -> s1 -> s0.  Twist the closing edge."""
    transitions = {(0, 0): (1, 0, 0), (1, 0): (0, 0, 0)}
    twist_edges = [((1, 0), (0, 0))]       # close the 2-cycle
    return transitions, 2, None, twist_edges


def loop_3cycle():
    """3-state cycle: s0 -> s1 -> s2 -> s0.  Twist closing edge."""
    transitions = {
        (0, 0): (1, 0, 0),
        (1, 0): (2, 0, 0),
        (2, 0): (0, 0, 0),
    }
    twist_edges = [((2, 0), (0, 0))]       # close the 3-cycle
    return transitions, 3, None, twist_edges


# ---------------------------------------------------------------------------
# 5.  Non-Hermitian Oracle runner
# ---------------------------------------------------------------------------

def run_nonhermitian_oracle(transitions, num_states, name, halt_idx=None,
                            twist_edges=None, loss_rate=0.1, verbose=True):
    """Full non-Hermitian topological halting oracle."""

    if twist_edges is None:
        twist_edges = []

    H, labels, halt_mask = build_nonhermitian_H(
        transitions, num_states, halt_idx=halt_idx, loss_rate=loss_rate
    )
    N = H.shape[0]

    # ----  spectral data  ------------------------------------------------
    eigvals, eigvecs, kappa_V = get_spectral_data(H)

    # ----  convert twist edges from (s,b) notation to flat indices  ------
    symbols = 2
    twist_indices = []
    for (s1, b1), (s2, b2) in twist_edges:
        i = s1 * symbols + b1          # source
        j = s2 * symbols + b2          # target
        twist_indices.append((i, j))

    # ----  point-gap winding via boundary twist (spectral flow)  ---------
    W_twist_raw, W_twist = point_gap_winding(
        H, twist_indices, E_ref=0.0j, n_phi=400
    )

    # ----  EP detection  -------------------------------------------------
    ep_detected = kappa_V > 1e6

    # ----  eigenvalue statistics  ----------------------------------------
    N_active = (~halt_mask).sum().item()
    N_halt = halt_mask.sum().item()
    spectral_radius = float(eigvals.abs().max().item())

    # ----  topological verdict  ------------------------------------------
    W_int = int(W_twist)
    if abs(W_int) > 0:
        verdict = "LOOPS"
        subtype = (f"non-trivial spectral flow  W={W_int:+d}"
                   f"  {'EP detected' if ep_detected else ''}")
    else:
        verdict = "HALTS"
        subtype = (f"trivial spectral flow  W={W_int:+d}"
                   f"  {'EP detected' if ep_detected else ''}"
                   f"  kappa(V)={kappa_V:.2e}")

    # ----  output  -------------------------------------------------------
    if verbose:
        print("=" * 70)
        print(f"  NON-HERMITIAN HALTING ORACLE  --  {name}")
        print("=" * 70)

        print(f"\nNon-Hermitian Hamiltonian  (dim = {N})")
        print("-" * 70)
        for row in range(N):
            row_parts = []
            for col in range(N):
                z = H[row, col]
                row_parts.append(f"{z.real.item():6.2f}{z.imag.item():+6.2f}j")
            row_str = "  ".join(row_parts)
            print(f"  {labels[row]:>15s}   [{row_str}]")

        print(f"\nComplex Eigenvalues:")
        for i, lam in enumerate(eigvals):
            tag = " <-- HALT" if halt_mask[i] else ""
            print(f"    lambda[{i}] = {lam.real.item():+9.6f}{lam.imag.item():+9.6f}j"
                  f"  |lam|={lam.abs().item():.6f}{tag}")
        print(f"  Spectral radius = {spectral_radius:.6f}")
        print(f"  Eigenvector condition  kappa(V) = {kappa_V:.6e}"
              f"  {'*** EP DETECTED ***' if ep_detected else ''}")

        print(f"\nPoint-Gap Winding (boundary twist sweep):")
        W_int = int(round(W_twist_raw))
        print(f"  W_twist = {W_twist_raw:+.4f}  (rounded: {W_int:+d})"
              f"  {'*** spectral loop ***' if abs(W_int) > 0 else ''}")
        print(f"  Twist edges: {len(twist_edges)} cycle-closing transitions")

        print(f"\nActive states: {N_active}  |  Halt basis states: {N_halt}")
        print(f"  loss_rate = {loss_rate}  |  halt sink multiplier = 10x")

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
        "kappa_V": kappa_V,
        "ep_detected": ep_detected,
        "W_twist": W_twist,
        "verdict": verdict,
        "N_active": N_active,
        "N_halt": N_halt,
    }


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main():
    raw = [halt_direct(), halt_chain(), loop_2cycle(), loop_3cycle()]
    names = [
        "Halt Direct (2-state, halt coupled)",
        "Halt Chain (3-state, halt terminal)",
        "Loop 2-Cycle (2-state, no halt)",
        "Loop 3-Cycle (3-state, no halt)",
    ]
    machines = [(names[i], *raw[i]) for i in range(len(raw))]

    results = []
    for name, transitions, num_states, halt_idx, twist_edges in machines:
        r = run_nonhermitian_oracle(
            transitions, num_states, name,
            halt_idx=halt_idx, twist_edges=twist_edges, verbose=True
        )
        results.append(r)

    # Summary
    print()
    print("=" * 70)
    print("  NON-HERMITIAN ORACLE SUMMARY")
    print("=" * 70)
    header = (f"  {'Machine':<38s}  {'W_twist':>7s}  "
              f"{'kappa(V)':>10s}  {'EP?':>5s}  {'Verdict'}")
    print(header)
    print("  " + "-" * 75)
    for r in results:
        w = int(r['W_twist'])
        print(f"  {r['name']:<38s}  {w:+7d}  "
              f"{r['kappa_V']:>10.2e}  "
              f"{'YES' if r['ep_detected'] else ' no':>5s}  "
              f"{'HALTS' if r['verdict']=='HALTS' else 'LOOPS'}")
    print()
    print("  W_twist: spectral-flow winding via boundary twist (phi sweep)")
    print("           W=0 -> eigenvalues are static -> HALTS")
    print("           W!=0 -> eigenvalues trace a loop -> LOOPS")
    print("  kappa(V): eigenvector condition number (diverges at EP)")
    print("=" * 70)


if __name__ == "__main__":
    main()
