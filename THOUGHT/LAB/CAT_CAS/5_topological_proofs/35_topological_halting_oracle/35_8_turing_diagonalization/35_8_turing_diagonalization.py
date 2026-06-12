"""
35_8_turing_diagonalization.py

Turing Diagonalization as a Topological Obstruction.

PHYSICS:  Turing's undecidability proof — "Does the machine that halts
on all looping machines halt?" — can be reframed as a topological
question:  is the winding number globally defined on the parameter
space of TM Hamiltonians?

CONSTRUCTION:
  Build a parameter-dependent Hamiltonian H(lambda, phi) that smoothly
  interpolates between halting (lam=0) and looping (lam>0):
    H = [[-iL,        gamma*lam*e^{i*phi}],
         [gamma*(1-lam),  -iL           ]]
  At lam=0 or lam=1: lower/upper triangular -> winding W=0 (HALT).
  At lam in (0,1): bidirectional coupling -> winding W=1 (LOOP).

  Measure Berry phase, Chern number, and eigenvector holonomy to
  determine whether the spectral bundle is trivial (Z) or obstructed
  (Z_2).  Honest result: the bundle IS trivial on this 2-parameter
  torus.  A true Godel obstruction requires a self-referential
  fixed-point singularity.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
from torch import linalg as LA

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  Godel Hamiltonian  H(lambda, phi)
# ---------------------------------------------------------------------------

def godel_hamiltonian(lam, phi, loss=0.1, gamma=1.0):
    """
    Parameter-dependent non-Hermitian 2x2 Hamiltonian.
    lam in [0,1]: 0=halt (forward-only), 0.5=Godel (symmetric), 1=halt (reverse-only)
    phi in [0,2pi): boundary twist on the forward edge.

    H = [[-iL,       gamma * lam * e^{i*phi}   ],
         [gamma*(1-lam),  -iL                   ]]

    At lam=0:    H[1][0]=gamma, H[0][1]=0 -> lower triangular -> HALT
    At lam=0.5:  H[1][0]=H[0][1]=gamma/2 -> symmetric -> GODEL POINT
    At lam=1:    H[1][0]=0, H[0][1]=gamma -> upper triangular -> HALT
    """
    H = torch.zeros((2, 2), dtype=torch.complex64)
    twist = torch.tensor(np.exp(1j * phi), dtype=torch.complex64)
    H[0, 0] = -1j * loss
    H[1, 1] = -1j * loss
    H[1, 0] = gamma * (1.0 - lam) + 0j         # reverse coupling
    H[0, 1] = gamma * lam * twist               # forward coupling (twisted)
    return H


# ---------------------------------------------------------------------------
# 2.  Winding number W(lambda) — detect spectral loop
# ---------------------------------------------------------------------------

def winding_lambda(lam, n_phi=200):
    """Compute W(lam) by sweeping phi at fixed lambda."""
    dets = torch.zeros(n_phi, dtype=torch.complex64)
    for k in range(n_phi):
        phi = 2.0 * np.pi * k / n_phi
        H = godel_hamiltonian(lam, phi)
        dets[k] = LA.det(H)
    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta + np.pi, 2.0 * np.pi) - np.pi
    return int(round(float(torch.sum(dtheta).item()) / (2.0 * np.pi)))


# ---------------------------------------------------------------------------
# 3.  Eigenvector holonomy (Berry phase)
# ---------------------------------------------------------------------------

def berry_phase_loop(lam, n_phi=200):
    """
    Compute the Berry phase accumulated by the dominant eigenvector
    of H(lam, phi) as phi sweeps [0, 2pi).

    Berry connection: A(phi) = -Im <v(phi)| d/dphi |v(phi)>
    Berry phase:      gamma = \oint A(phi) dphi

    For a Mobius strip: gamma = pi (mod 2pi) -> non-trivial Z_2.
    For a trivial bundle: gamma = 0 -> trivial.
    """
    phi_vals = torch.linspace(0.0, 2.0 * np.pi, n_phi)
    eigvecs_phi = torch.zeros((n_phi, 2), dtype=torch.complex64)
    eigvals_phi = torch.zeros((n_phi, 2), dtype=torch.complex64)

    for k, phi in enumerate(phi_vals):
        H = godel_hamiltonian(lam, phi.item())
        ev, eV = LA.eig(H)
        # Sort by |eigenvalue| to track the same eigenvector
        order = ev.abs().argsort(descending=True)
        eigvals_phi[k] = ev[order]
        eigvecs_phi[k] = eV[:, order[0]]

    # Ensure phase continuity (fix branch cuts)
    for k in range(1, n_phi):
        # Align phase with previous eigenvector
        overlap = eigvecs_phi[k - 1].conj().dot(eigvecs_phi[k])
        eigvecs_phi[k] = eigvecs_phi[k] * (overlap / (overlap.abs() + 1e-30))

    # Compute Berry connection via discrete differences
    berry_phase = 0.0
    for k in range(n_phi):
        k_next = (k + 1) % n_phi
        overlap = eigvecs_phi[k].conj().dot(eigvecs_phi[k_next])
        berry_phase -= float(torch.angle(overlap).item())

    return berry_phase % (2.0 * np.pi)


# ---------------------------------------------------------------------------
# 4.  Chern number on the (lam, phi) torus
# ---------------------------------------------------------------------------

def chern_number(n_lam=20, n_phi=20):
    """
    Compute the first Chern number via discrete Berry curvature.

    For each plaquette on the (lam, phi) grid:
      F = arg( <v_ij | v_{i+1,j}> * <v_{i+1,j} | v_{i+1,j+1}>
             * <v_{i+1,j+1} | v_{i,j+1}> * <v_{i,j+1} | v_ij> )
      Chern = (1/2pi) sum_plaquettes F
    """
    lam_vals = torch.linspace(0.0, 1.0, n_lam)
    phi_vals = torch.linspace(0.0, 2.0 * np.pi, n_phi)

    # Eigenvectors at each grid point (track by eigenvalue magnitude)
    evecs = torch.zeros((n_lam, n_phi, 2), dtype=torch.complex64)

    for i, lam in enumerate(lam_vals):
        for j, phi in enumerate(phi_vals):
            H = godel_hamiltonian(lam.item(), phi.item())
            ev, eV = LA.eig(H)
            dominant = ev.abs().argmax()
            evecs[i, j] = eV[:, dominant]

    # Phase-align across the grid (fix branch cuts)
    for i in range(n_lam):
        for j in range(n_phi):
            if j < n_phi - 1:
                ov = evecs[i, j].conj().dot(evecs[i, j + 1])
                evecs[i, j + 1] = evecs[i, j + 1] * (ov / (ov.abs() + 1e-30))
    for i in range(n_lam - 1):
        for j in range(n_phi):
            ov = evecs[i, j].conj().dot(evecs[i + 1, j])
            evecs[i + 1, j] = evecs[i + 1, j] * (ov / (ov.abs() + 1e-30))

    # Compute U(1) link variables from phase-aligned eigenvectors
    U_lam = torch.zeros((n_lam - 1, n_phi), dtype=torch.float64)
    U_phi = torch.zeros((n_lam, n_phi - 1), dtype=torch.float64)
    for i in range(n_lam):
        for j in range(n_phi):
            if j < n_phi - 1:
                ov = evecs[i, j].conj().dot(evecs[i, j + 1])
                U_phi[i, j] = float(torch.angle(ov).item())
            if i < n_lam - 1:
                ov = evecs[i, j].conj().dot(evecs[i + 1, j])
                U_lam[i, j] = float(torch.angle(ov).item())

    # Berry curvature: plaquette flux
    chern = 0.0
    for i in range(n_lam - 1):
        for j in range(n_phi - 1):
            F = (U_lam[i, j] + U_phi[i + 1, j] -
                 U_lam[i, j + 1] - U_phi[i, j])
            F = (F + np.pi) % (2.0 * np.pi) - np.pi
            chern += F
    return chern / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# 5.  Winding transition analysis
# ---------------------------------------------------------------------------

def winding_transition():
    """Sweep lambda to find the winding-number transition (Godel point)."""
    print("=" * 70)
    print("  WINDING TRANSITION — W(lambda) across the Godel point")
    print("=" * 70)
    print(f"  {'lambda':>8s}  {'W':>4s}  {'Phase'}")
    print("  " + "-" * 30)

    lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49,
                0.5, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    for lam in lam_vals:
        W = winding_lambda(lam)
        phase = "HALT" if W == 0 else "LOOP"
        marker = " <-- GODEL POINT" if abs(lam - 0.5) < 0.01 else ""
        print(f"  {lam:8.3f}  {W:4d}  {phase}{marker}")

    print()
    print(f"  At lambda=0.5 (Godel point): W=1, eigenvalues non-degenerate")
    print(f"  Winding transitions at boundaries only: lam=0 -> W=0, lam=1 -> W=0")
    print(f"  For 0 < lam < 1: W=1 (spectral loop from bidirectional coupling)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main():
    # Winding transition (Godel point)
    winding_transition()

    # Berry phase at key lambda values
    print(f"\n{'=' * 70}")
    print("  BERRY PHASE / EIGENVECTOR HOLONOMY")
    print(f"{'=' * 70}")
    print(f"  {'lambda':>8s}  {'Berry phase':>14s}  {'Topology'}")
    print("  " + "-" * 45)

    for lam in [0.0, 0.3, 0.49, 0.5, 0.51, 0.7, 1.0]:
        gamma = berry_phase_loop(lam, n_phi=200)
        if abs(gamma - 2.0*np.pi) < 0.3 or gamma < 0.2:
            topo = "TRIVIAL"
        else:
            topo = f"phase={gamma:.2f}"
        marker = " <-- GODEL" if abs(lam - 0.5) < 0.01 else ""
        print(f"  {lam:8.3f}  {gamma:14.6f}  {topo}{marker}")

    print()
    print("  Berry phase = 2pi (trivial) -> Godel point is topologically trivial")
    print("  Berry phase != 0, 2pi at intermediate lam -> geometric phase from")
    print("  the spectral loop, not a Z_2 obstruction")
    print(f"{'=' * 70}")

    # Chern number
    print()
    C = chern_number(n_lam=30, n_phi=30)
    print(f"{'=' * 70}")
    print(f"  CHERN NUMBER on (lambda, phi) torus:  C = {C:.4f}")
    print(f"  {'Half-integer -> Z_2 obstruction (Mobius strip)' if abs(C - 0.5) < 0.2 else 'Integer -> Z classification'}")

    # Conclusion
    print(f"\n  The Chern number C = {C:.4f}.  ", end="")
    if abs(C) < 0.1:
        print("The eigenvector bundle is TRIVIAL")
        print(f"  — winding IS globally defined on this parameter manifold.")
        print(f"  The Godel obstruction requires a genuinely self-referential")
        print(f"  Hamiltonian where the TM's own description enters the")
        print(f"  parameter space, creating a fixed-point singularity.")
    elif abs(abs(C) - 0.5) < 0.2:
        print("Z_2 obstruction detected (Mobius strip)")
    else:
        print(f"Integer Chern (Z classification)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
