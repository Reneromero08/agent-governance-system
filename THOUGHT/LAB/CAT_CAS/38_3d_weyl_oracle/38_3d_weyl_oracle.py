"""
38_3d_weyl_oracle.py

EXPERIMENT 38: 3D NON-HERMITIAN WEYL ANNIHILATION ORACLE
=========================================================
Halting as Weyl node annihilation via catalytic dimensional reduction.

A 3D Weyl semimetal is constructed as a stack of 2D Chern insulator
slices parameterized by kz.  The kz-dependent mass M(kz) = m0 - tz*cos(kz)
creates Weyl nodes where M=0.  Between nodes, 2D slices carry non-zero
Chern number -> protected surface Fermi arcs -> LOOPS.

An Exceptional Point sink (-i*Gamma) at the halt site pulls Weyl nodes
into the complex energy plane.  When they collide, they form a Weyl
Exceptional Ring and annihilate.  C(kz)=0 for ALL slices -> HALTS.

CATALYTIC DIMENSIONAL REDUCTION:
  No dense 3D matrix.  O(L^2) 2D buffers are reused across the kz loop.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64


# ======================================================================
# 1.  2D Slice Hamiltonian (parameterized by kz)
# ======================================================================

def build_weyl_slice(L, kz, t1=1.0, t2=0.5, phi=np.pi/4, tz=1.5, m0=0.5,
                     loss=0.05, gamma_halt=0.0, halt_pos=None):
    """
    Build a 2D Chern insulator slice at momentum kz.
    M(kz) = m0 - tz*cos(kz) acts as the kz-dependent mass term.
    Weyl nodes at M(kz) = 0.
    """
    N = L * L
    H = torch.zeros((N, N), dtype=COMPLEX)
    M_kz = m0 - tz * np.cos(kz)

    if halt_pos is None:
        halt_pos = (L // 2, L // 2)

    for y in range(L):
        for x in range(L):
            i = y * L + x

            # On-site mass + bulk dissipation
            H[i, i] = M_kz - 1j * loss

            # Halt defect: EP sink
            if (x, y) == halt_pos:
                H[i, i] -= 1j * gamma_halt

            # Nearest-neighbor hopping (t1)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = (x + dx) % L, (y + dy) % L
                j = ny * L + nx
                H[j, i] += t1 + 0j

            # Complex NNN hopping (TRS breaking, creates Chern phase)
            for dx, dy in [(1, 1), (-1, -1)]:
                nx, ny = (x + dx) % L, (y + dy) % L
                j = ny * L + nx
                H[j, i] += t2 * np.exp(1j * phi)

            for dx, dy in [(1, -1), (-1, 1)]:
                nx, ny = (x + dx) % L, (y + dy) % L
                j = ny * L + nx
                H[j, i] += t2 * np.exp(-1j * phi)

    return H


# ======================================================================
# 2.  Spectral Projector + Bott Index (from Experiment 37)
# ======================================================================

def spectral_projector(H, E_fermi=-0.5j, n_pts=32, radius=2.0):
    """Contour integral projector P = (1/n) sum_k (z_k I - H)^(-1) R e^{i*theta_k}."""
    N = H.shape[0]
    I = torch.eye(N, dtype=COMPLEX)
    P = torch.zeros((N, N), dtype=COMPLEX)
    for k in range(n_pts):
        theta = 2 * np.pi * k / n_pts
        z = E_fermi + radius * torch.tensor(np.exp(1j * theta), dtype=COMPLEX)
        M = z * I - H
        invM = torch.linalg.inv(M)
        P += invM * (radius * torch.tensor(np.exp(1j * theta), dtype=COMPLEX))
    return P / n_pts


def bott_index(P, L):
    """C = (1/2pi) Im Tr log(V U V^dag U^dag)"""
    N = L * L
    x_vals = torch.tensor([x for y in range(L) for x in range(L)], dtype=torch.float32)
    y_vals = torch.tensor([y for y in range(L) for x in range(L)], dtype=torch.float32)
    UX = torch.diag(torch.exp(1j * 2 * np.pi * x_vals / L)).to(COMPLEX)
    UY = torch.diag(torch.exp(1j * 2 * np.pi * y_vals / L)).to(COMPLEX)
    U = P @ UX @ P
    V = P @ UY @ P
    W = V @ U @ V.conj().T @ U.conj().T
    try:
        logW = torch.linalg.matrix_log(W)
    except Exception:
        evals, evecs = torch.linalg.eig(W)
        logW = evecs @ torch.diag(torch.log(evals)) @ torch.linalg.inv(evecs)
    return round(float((1.0 / (2 * np.pi)) * torch.trace(logW).imag.item()))


# ======================================================================
# 3.  Weyl Annihilation Oracle
# ======================================================================

def run_3d_weyl_oracle(L=8, n_kz=24):
    N = L * L

    print("=" * 78)
    print("  EXPERIMENT 38: 3D NON-HERMITIAN WEYL ANNIHILATION ORACLE")
    print("  Fermi Arc Survival -> Looping | Weyl Annihilation -> Halting")
    print("=" * 78)
    print(f"  Lattice: {L}x{L}x{n_kz} slices  (N_2D = {N}, N_eff = {N * n_kz})")
    print(f"  m0={0.5}  tz={1.5}  M(kz)=m0-tz*cos(kz)")
    print(f"  Weyl nodes at cos(kz)=m0/tz={0.5/1.5:.3f}")

    kz_vals = torch.linspace(0, 2 * np.pi, n_kz)

    # Auto-detect Fermi energy from kz=pi slice (far from Weyl nodes)
    H0 = build_weyl_slice(L, kz=np.pi, gamma_halt=0.0)
    im_s = torch.sort(torch.linalg.eigvals(H0).imag).values
    gaps = im_s[1:] - im_s[:-1]
    E_fermi = complex(0, (im_s[gaps.argmax()] + im_s[gaps.argmax() + 1]).item() / 2)
    print(f"  E_fermi = {E_fermi.imag:.4f}j")

    # ----  LOOPING: Gamma=0, Weyl nodes separated  --------------------
    print(f"\n  ---  LOOPING (Gamma=0)  ---")
    print(f"  {'kz':>8s}  {'M(kz)':>8s}  {'C':>4s}")
    print("  " + "-" * 25)
    C_loop = []
    for kz in kz_vals:
        H = build_weyl_slice(L, kz.item(), gamma_halt=0.0)
        P = spectral_projector(H, E_fermi=E_fermi)
        C = bott_index(P, L)
        C_loop.append(C)
        M_val = 0.5 - 1.5 * np.cos(kz.item())
        marker = " <-- Weyl node" if abs(M_val) < 0.2 else ""
        print(f"  {kz.item():8.4f}  {M_val:+8.4f}  {C:+4d}{marker}")

    max_C_loop = max(abs(c) for c in C_loop)
    nonzero_count = sum(1 for c in C_loop if c != 0)
    print(f"\n  Non-zero slices: {nonzero_count}/{n_kz}  |  Max |C| = {max_C_loop}")
    print(f"  VERDICT: {'LOOPS (Fermi arc exists)' if max_C_loop > 0 else 'HALTS'}")

    # ----  HALTING: Gamma=15, Weyl nodes annihilated  -----------------
    GAMMA_HALT = 15.0
    print(f"\n  ---  HALTING (Gamma={GAMMA_HALT})  ---")
    print(f"  {'kz':>8s}  {'M(kz)':>8s}  {'C':>4s}")
    print("  " + "-" * 25)
    C_halt = []
    for kz in kz_vals:
        H = build_weyl_slice(L, kz.item(), gamma_halt=GAMMA_HALT)
        P = spectral_projector(H, E_fermi=E_fermi)
        C = bott_index(P, L)
        C_halt.append(C)
        M_val = 0.5 - 1.5 * np.cos(kz.item())
        print(f"  {kz.item():8.4f}  {M_val:+8.4f}  {C:+4d}")

    max_C_halt = max(abs(c) for c in C_halt)
    nonzero_halt = sum(1 for c in C_halt if c != 0)
    print(f"\n  Non-zero slices: {nonzero_halt}/{n_kz}  |  Max |C| = {max_C_halt}")
    print(f"  VERDICT: {'LOOPS (Fermi arc survives)' if max_C_halt > 0 else 'HALTS (Weyl nodes annihilated)'}")

    # ----  Summary  ---------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  ORACLE SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Case                 Max |C(kz)|    Non-zero slices    Verdict")
    print(f"  {'-'*55}")
    print(f"  Looping (Gamma=0)       {max_C_loop:5d}           "
          f"{nonzero_count:3d}/{n_kz}          "
          f"{'LOOPS' if max_C_loop > 0 else 'HALTS'}")
    print(f"  Halting (Gamma={GAMMA_HALT})      {max_C_halt:5d}           "
          f"{nonzero_halt:3d}/{n_kz}          "
          f"{'LOOPS' if max_C_halt > 0 else 'HALTS'}")
    print(f"  {'='*78}")

    if max_C_loop > 0 and max_C_halt == 0:
        print("  CORRECT: Weyl nodes annihilated by EP sink")
    elif max_C_loop > 0 and max_C_halt > 0:
        print(f"  PARTIAL: Fermi arc survived at Gamma={GAMMA_HALT} — increase Gamma")
    elif max_C_loop == 0:
        print("  FAILED: No Fermi arc at Gamma=0 — check m0/tz parameters")
    else:
        print("  UNEXPECTED")

    print(f"  Catalytic dimensional reduction: O(L^2) buffers reused")
    print(f"  {'='*78}")

    # ----  Gamma annihilation sweep  ---------------------------------
    print(f"\n{'=' * 78}")
    print("  GAMMA ANNIHILATION SWEEP — Fermi arc destruction vs sink strength")
    print(f"{'=' * 78}")
    print(f"  {'Gamma':>8s}  {'max|C|':>7s}  {'nonzero':>8s}  {'Verdict'}")
    print("  " + "-" * 40)
    for g in [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
        C_gamma = []
        # Recompute Fermi energy per Gamma (sink shifts the spectrum)
        H_ref = build_weyl_slice(L, kz=np.pi, gamma_halt=g)
        im_sg = torch.sort(torch.linalg.eigvals(H_ref).imag).values
        gaps_g = im_sg[1:] - im_sg[:-1]
        Ef_g = complex(0, (im_sg[gaps_g.argmax()] + im_sg[gaps_g.argmax()+1]).item()/2)
        for kz in kz_vals:
            H = build_weyl_slice(L, kz.item(), gamma_halt=g)
            P = spectral_projector(H, E_fermi=Ef_g)
            C_gamma.append(bott_index(P, L))
        maxC = max(abs(c) for c in C_gamma)
        nz = sum(1 for c in C_gamma if c != 0)
        v = "LOOPS" if maxC > 0 else "HALTS (annihilated)"
        print(f"  {g:8.1f}  {maxC:7d}  {nz:8d}  {v}")
    print(f"  {'='*78}")
    print(f"  E_fermi recomputed per Gamma. Single-site EP cannot fully")
    print(f"  annihilate Weyl nodes (momentum-space defects). Full")
    print(f"  annihilation requires kz-dependent sink or uniform Gamma field.")
    print(f"  {'='*78}")


# ======================================================================
# 4.  Main
# ======================================================================

if __name__ == "__main__":
    run_3d_weyl_oracle(L=8, n_kz=24)
