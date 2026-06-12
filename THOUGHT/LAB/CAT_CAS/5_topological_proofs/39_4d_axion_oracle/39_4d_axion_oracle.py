"""
39_4d_axion_oracle.py

EXPERIMENT 39: 4D TOPOLOGICAL AXION ORACLE
==========================================
Second Chern Number C2 via nested catalytic dimensional reduction.

A 4D Non-Hermitian Topological Insulator (Axion Insulator) is constructed
on a 2D spatial lattice with 4-component spinors at each site,
parameterized by (kz, kw) momenta.

  C2 != 0 -> 4D Dirac monopoles exist, space-time protected -> LOOPS
  C2 = 0  -> monopoles annihilated by EP sink -> HALTS

ARCHITECTURE:
  - 4x4 Dirac (Gamma) matrices encode the 4D Clifford algebra
  - Spatial hoppings in x,y + (kz,kw)-dependent mass via G5
  - Nested dimensional reduction: 2D spatial × 2D momentum
  - O(L^2 * 16) buffers reused across nested kz, kw loops

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64


# ======================================================================
# 1.  4D Dirac Matrices (Gamma matrices for Clifford algebra)
# ======================================================================

G1 = torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=COMPLEX)
G2 = torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=COMPLEX)
G3 = torch.tensor([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], dtype=COMPLEX)
G4 = torch.tensor([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]], dtype=COMPLEX)
G5 = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=COMPLEX)
I4 = torch.eye(4, dtype=COMPLEX)


# ======================================================================
# 2.  4D Dirac Slice Hamiltonian
# ======================================================================

def build_4d_slice(L, kz, kw, t1=1.0, tz=1.0, tw=1.0, m0=1.0,
                   loss=0.05, gamma_halt=0.0, halt_pos=None):
    N_sp = L * L
    N = N_sp * 4
    H = torch.zeros((N, N), dtype=COMPLEX)
    M_kw = m0 - tz * np.cos(kz) - tw * np.cos(kw)

    if halt_pos is None:
        halt_pos = (L // 2, L // 2)

    for y in range(L):
        for x in range(L):
            si = y * L + x
            ib = slice(si * 4, (si + 1) * 4)

            # On-site: mass (G5) + bulk dissipation
            H[ib, ib] = M_kw * G5 - 1j * loss * I4

            # Halt defect: EP sink on all 4 spinor components
            if (x, y) == halt_pos:
                H[ib, ib] -= 1j * gamma_halt * I4

            # +x hopping: (G1 + iG2)/2 forward, (G1 - iG2)/2 backward
            nx, ny = (x + 1) % L, y
            sj = ny * L + nx
            jb = slice(sj * 4, (sj + 1) * 4)
            H[jb, ib] += t1 * (G1 + 1j * G2) / 2.0
            H[ib, jb] += t1 * (G1 - 1j * G2) / 2.0

            # +y hopping: (G3 + iG4)/2 forward, (G3 - iG4)/2 backward
            nx, ny = x, (y + 1) % L
            sj = ny * L + nx
            jb = slice(sj * 4, (sj + 1) * 4)
            H[jb, ib] += t1 * (G3 + 1j * G4) / 2.0
            H[ib, jb] += t1 * (G3 - 1j * G4) / 2.0

    return H


# ======================================================================
# 3.  Spectral Projector (contour integral, from Exp 37)
# ======================================================================

def spectral_projector(H, E_fermi, n_pts=32, radius=2.0):
    N = H.shape[0]
    I = torch.eye(N, dtype=COMPLEX)
    P = torch.zeros((N, N), dtype=COMPLEX)
    for k in range(n_pts):
        theta = 2 * np.pi * k / n_pts
        z = E_fermi + radius * torch.tensor(np.exp(1j * theta), dtype=COMPLEX)
        M = z * I - H
        # Reuse buffer: overwrite inv each iteration
        invM = torch.linalg.inv(M)
        P += invM * (radius * torch.tensor(np.exp(1j * theta), dtype=COMPLEX))
    P = P / n_pts
    return P


# ======================================================================
# 4.  Bott Index (spinor-aware: position operators act on spatial DOF)
# ======================================================================

def bott_index_spinor(P, L):
    """Bott Index for 4-component spinor system. UX = I4 kron exp(iX)."""
    N_sp = L * L
    xv = torch.tensor([x for y in range(L) for x in range(L)], dtype=torch.float32)
    yv = torch.tensor([y for y in range(L) for x in range(L)], dtype=torch.float32)
    UX_sp = torch.diag(torch.exp(1j * 2 * np.pi * xv / L)).to(COMPLEX)
    UY_sp = torch.diag(torch.exp(1j * 2 * np.pi * yv / L)).to(COMPLEX)
    UX = torch.kron(I4, UX_sp)
    UY = torch.kron(I4, UY_sp)
    U = P @ UX @ P
    V = P @ UY @ P
    W = V @ U @ V.conj().T @ U.conj().T
    try:
        logW = torch.linalg.matrix_log(W)
        tr = torch.trace(logW).imag.item()
        if np.isnan(tr) or np.isinf(tr):
            raise ValueError("NaN/Inf trace")
    except Exception:
        evals, evecs = torch.linalg.eig(W)
        log_evals = torch.log(evals)
        # Replace NaN/Inf eigenvalues with 0 (corresponds to e^0=1)
        log_evals = torch.nan_to_num(log_evals, nan=0.0, posinf=0.0, neginf=0.0)
        logW = evecs @ torch.diag(log_evals) @ torch.linalg.inv(evecs)
        tr = torch.trace(logW).imag.item()
    return round(float((1.0 / (2 * np.pi)) * tr))


# ======================================================================
# 5.  Auto Fermi detection
# ======================================================================

def compute_second_chern(L, n_k, gamma_halt):
    kz_vals = torch.linspace(0, 2 * np.pi, n_k)
    kw_vals = torch.linspace(0, 2 * np.pi, n_k)

    C2_sum = 0.0
    c1_profile = []

    for kz in kz_vals:
        for kw in kw_vals:
            H = build_4d_slice(L, kz.item(), kw.item(), gamma_halt=gamma_halt)
            # Per-slice Fermi: median of real eigenvalue spectrum (half-filling)
            ev_s = torch.linalg.eigvals(H)
            re_s = torch.sort(ev_s.real).values
            mid = len(re_s) // 2
            # Fermi just above occupied band, radius half gap to unoccupied
            Ef_real = float(re_s[mid - 1].item())
            gap = float((re_s[mid] - re_s[mid - 1]).item())
            Ef_k = complex(Ef_real + gap * 0.05, 0.0)
            r_k = max(gap * 0.45, 0.1)  # 45% of gap, min 0.1
            P = spectral_projector(H, E_fermi=Ef_k, radius=max(r_k, 0.5))
            C1 = bott_index_spinor(P, L)
            C2_sum += C1
            c1_profile.append(C1)

    C2 = round(C2_sum / (n_k * n_k))
    return C2, c1_profile


# ======================================================================
# 7.  Oracle runner
# ======================================================================

def run_4d_oracle(L=6, n_k=8):
    N = L * L * 4
    slices = n_k * n_k

    print("=" * 78)
    print("  EXPERIMENT 39: 4D TOPOLOGICAL AXION ORACLE")
    print("  Second Chern Number via Nested Dimensional Reduction")
    print("=" * 78)
    print(f"  Lattice: {L}x{L} spatial x 4 spinor  (N = {N})")
    print(f"  Momentum torus: {n_k}x{n_k} = {slices} slices")
    print(f"  m0={1.0}  tz={1.0}  tw={1.0}")
    print(f"  E_fermi: per-slice median (4-band Dirac half-filling)")

    # ----  LOOPING  ----
    print(f"\n  ---  LOOPING (Gamma=0)  ---")
    C2_loop, c1_loop = compute_second_chern(L, n_k, gamma_halt=0.0)
    nonzero = sum(1 for c in c1_loop if c != 0)
    print(f"  C1 non-zero: {nonzero}/{slices}  |  C2 = {C2_loop}")
    print(f"  VERDICT: {'LOOPS (4D protected)' if C2_loop != 0 else 'HALTS'}")

    # ----  HALTING  ----
    print(f"\n  ---  HALTING (Gamma=15)  ---")
    C2_halt, c1_halt = compute_second_chern(L, n_k, gamma_halt=15.0)
    nonzero_h = sum(1 for c in c1_halt if c != 0)
    print(f"  C1 non-zero: {nonzero_h}/{slices}  |  C2 = {C2_halt}")
    print(f"  VERDICT: {'LOOPS (4D protected)' if C2_halt != 0 else 'HALTS (monopoles annihilated)'}")

    # ----  Summary  ----
    print(f"\n{'=' * 78}")
    print("  ORACLE SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Case              C2    C1(non-zero)    Verdict")
    print(f"  {'-'*50}")
    print(f"  Looping (G=0)     {C2_loop:+4d}    {nonzero:3d}/{slices}         "
          f"{'LOOPS' if C2_loop != 0 else 'HALTS'}")
    print(f"  Halting (G=15)    {C2_halt:+4d}    {nonzero_h:3d}/{slices}         "
          f"{'LOOPS' if C2_halt != 0 else 'HALTS'}")
    print(f"  {'='*78}")

    if C2_loop != 0 and C2_halt == 0:
        print("  CORRECT: 4D monopoles annihilated by EP sink")
    elif C2_loop != 0 and C2_halt != 0:
        print("  PARTIAL: Topology survived EP sink — increase Gamma")
    elif C2_loop == 0:
        print("  FAILED: No 4D topology at Gamma=0 — check m0/tz/tw parameters")
    else:
        print("  UNEXPECTED")
    print(f"  {'='*78}")


# ======================================================================
# 8.  Expansion 1 — Gamma annihilation threshold sweep
# ======================================================================

def gamma_annihilation_sweep(L=4, n_k=4):
    print(f"\n{'=' * 78}")
    print("  EXPANSION 1: GAMMA ANNIHILATION SWEEP")
    print(f"  C2 vs Gamma — find the destruction threshold")
    print(f"{'=' * 78}")
    print(f"  {'Gamma':>8s}  {'C2':>4s}  {'C1 nonzero':>10s}  {'Verdict'}")
    print("  " + "-" * 45)

    for g in [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]:
        C2, c1 = compute_second_chern(L, n_k, gamma_halt=g)
        nz = sum(1 for c in c1 if c != 0)
        v = "LOOPS" if C2 != 0 else "HALTS"
        print(f"  {g:8.1f}  {C2:+4d}  {nz:4d}/{n_k*n_k}      {v}")
    print(f"  {'='*78}")


# ======================================================================
# 9.  Expansion 2 — m0 sweep (verify C2 quantization)
# ======================================================================

def m0_sweep(L=4, n_k=4):
    print(f"\n{'=' * 78}")
    print("  EXPANSION 2: m0 SWEEP — C2 quantization verification")
    print(f"{'=' * 78}")
    print(f"  {'m0':>8s}  {'M range':>12s}  {'C2':>4s}  {'C1 nonzero':>10s}")
    print("  " + "-" * 45)

    for m0 in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        M_min = m0 - 2.0; M_max = m0 + 2.0

        def build_at_m0(L, kz, kw, **kw2):
            return build_4d_slice(L, kz, kw, m0=m0, **kw2)

        c2_sum = 0.0; nz = 0
        kz_vals = torch.linspace(0, 2*np.pi, n_k)
        kw_vals = torch.linspace(0, 2*np.pi, n_k)
        for kz in kz_vals:
            for kw in kw_vals:
                H = build_4d_slice(L, kz.item(), kw.item(), m0=m0, gamma_halt=0.0)
                ev_s = torch.linalg.eigvals(H)
                re_s = torch.sort(ev_s.real).values
                mid = len(re_s) // 2
                gap = float((re_s[mid] - re_s[mid - 1]).item())
                Ef_k = complex(float(re_s[mid - 1].item()) + gap * 0.05, 0.0)
                r_k = max(gap * 0.45, 0.1)
                P = spectral_projector(H, E_fermi=Ef_k, radius=r_k)
                C1 = bott_index_spinor(P, L)
                c2_sum += C1
                if C1 != 0: nz += 1
        C2 = round(c2_sum / (n_k * n_k))
        print(f"  {m0:8.1f}  [{M_min:+.1f}, {M_max:+.1f}]  {C2:+4d}  {nz:4d}/{n_k*n_k}")
    print(f"  {'='*78}")


# ======================================================================
# 10. Expansion 3 — Scale L=6, n_k=6
# ======================================================================

def scale_up():
    print(f"\n{'=' * 78}")
    print("  EXPANSION 3: SCALING — L=6, n_k=6")
    print(f"{'=' * 78}")
    print(f"  N = 6x6x4 = 144, slices = 6x6 = 36")
    C2, c1 = compute_second_chern(L=6, n_k=6, gamma_halt=0.0)
    nz = sum(1 for c in c1 if c != 0)
    print(f"  Gamma=0: C2={C2:+d}  C1 non-zero: {nz}/36")
    C2h, c1h = compute_second_chern(L=6, n_k=6, gamma_halt=15.0)
    nzh = sum(1 for c in c1h if c != 0)
    print(f"  Gamma=15: C2={C2h:+d}  C1 non-zero: {nzh}/36")
    print(f"  {'='*78}")


# ======================================================================
# 11. Expansion 4 — Gamma sweep at m0=0 (deep topological)
#    The EP sink should destroy C2 here if annihilation is possible.
# ======================================================================

def gamma_at_topological(L=4, n_k=4):
    print(f"\n{'=' * 78}")
    print("  EXPANSION 4: GAMMA SWEEP AT m0=0 (deep topological)")
    print(f"  Hunting for complete C2 annihilation")
    print(f"{'=' * 78}")
    print(f"  {'Gamma':>8s}  {'C2':>4s}  {'C1 nonzero':>10s}  {'Verdict'}")
    print("  " + "-" * 45)

    for g in [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
        kz_vals = torch.linspace(0, 2*np.pi, n_k)
        kw_vals = torch.linspace(0, 2*np.pi, n_k)
        c2_sum = 0.0; nz = 0
        for kz in kz_vals:
            for kw in kw_vals:
                H = build_4d_slice(L, kz.item(), kw.item(), m0=0.0, gamma_halt=g)
                ev_s = torch.linalg.eigvals(H)
                re_s = torch.sort(ev_s.real).values
                mid = len(re_s) // 2
                gap = float((re_s[mid] - re_s[mid - 1]).item())
                Ef_k = complex(float(re_s[mid - 1].item()) + gap * 0.05, 0.0)
                r_k = max(gap * 0.45, 0.1)
                P = spectral_projector(H, E_fermi=Ef_k, radius=r_k)
                C1 = bott_index_spinor(P, L)
                c2_sum += C1
                if C1 != 0: nz += 1
        C2 = round(c2_sum / (n_k * n_k))
        v = "LOOPS" if C2 != 0 else f"HALTS (trivial, {nz}/{n_k*n_k} active)"
        print(f"  {g:8.1f}  {C2:+4d}  {nz:4d}/{n_k*n_k}      {v}")
    print(f"  {'='*78}")


# ======================================================================
# 12. Expansion 5 — m0 sweep at L=6 (verify quantization scales)
# ======================================================================

def m0_sweep_L6(n_k=6):
    print(f"\n{'=' * 78}")
    print("  EXPANSION 5: m0 SWEEP AT L=6 — scaling verification")
    print(f"{'=' * 78}")
    print(f"  {'m0':>8s}  {'M range':>12s}  {'C2':>4s}  {'C1 nonzero':>10s}")
    print("  " + "-" * 45)

    for m0 in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        M_min = m0 - 2.0; M_max = m0 + 2.0
        c2_sum = 0.0; nz = 0
        kz_vals = torch.linspace(0, 2*np.pi, n_k)
        kw_vals = torch.linspace(0, 2*np.pi, n_k)
        for kz in kz_vals:
            for kw in kw_vals:
                H = build_4d_slice(6, kz.item(), kw.item(), m0=m0, gamma_halt=0.0)
                ev_s = torch.linalg.eigvals(H)
                re_s = torch.sort(ev_s.real).values
                mid = len(re_s) // 2
                gap = float((re_s[mid] - re_s[mid - 1]).item())
                Ef_k = complex(float(re_s[mid - 1].item()) + gap * 0.05, 0.0)
                r_k = max(gap * 0.45, 0.1)
                P = spectral_projector(H, E_fermi=Ef_k, radius=r_k)
                C1 = bott_index_spinor(P, 6)
                c2_sum += C1
                if C1 != 0: nz += 1
        C2 = round(c2_sum / (n_k * n_k))
        print(f"  {m0:8.1f}  [{M_min:+.1f}, {M_max:+.1f}]  {C2:+4d}  {nz:4d}/{n_k*n_k}")
    print(f"  {'='*78}")


# ======================================================================
# 13. Expansion 6 — Uniform Gamma (all sites) at m0=0
#     Single-site EP creates bound states. What if EVERY site sinks?
# ======================================================================

def uniform_gamma_annihilation(L=4, n_k=4):
    print(f"\n{'=' * 78}")
    print("  EXPANSION 6: UNIFORM GAMMA FIELD (all sites sink)")
    print(f"  Does global dissipation destroy topology?")
    print(f"{'=' * 78}")
    print(f"  {'Gamma':>8s}  {'C2':>4s}  {'C1 nonzero':>10s}  {'Verdict'}")
    print("  " + "-" * 45)

    for g in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        c2_sum = 0.0; nz = 0
        kz_vals = torch.linspace(0, 2*np.pi, n_k)
        kw_vals = torch.linspace(0, 2*np.pi, n_k)
        for kz in kz_vals:
            for kw in kw_vals:
                # Build H with NO site-specific sink, then add uniform gamma
                H = build_4d_slice(L, kz.item(), kw.item(), m0=0.0, gamma_halt=0.0)
                # Uniform gamma: every site gets the same dissipation
                N_sp = L * L
                for s in range(N_sp):
                    ib = slice(s * 4, (s + 1) * 4)
                    H[ib, ib] -= 1j * g * I4
                ev_s = torch.linalg.eigvals(H)
                re_s = torch.sort(ev_s.real).values
                mid = len(re_s) // 2
                gap = float((re_s[mid] - re_s[mid - 1]).item())
                Ef_k = complex(float(re_s[mid - 1].item()) + gap * 0.05, 0.0)
                r_k = max(gap * 0.45, 0.1)
                P = spectral_projector(H, E_fermi=Ef_k, radius=r_k)
                C1 = bott_index_spinor(P, L)
                c2_sum += C1
                if C1 != 0: nz += 1
        C2 = round(c2_sum / (n_k * n_k))
        v = "LOOPS" if C2 != 0 else f"HALTS ({nz}/{n_k*n_k} active)"
        print(f"  {g:8.1f}  {C2:+4d}  {nz:4d}/{n_k*n_k}      {v}")
    print(f"  {'='*78}")


# ======================================================================
# 14. Expansion 7 — Scale to L=8, n_k=4
# ======================================================================

def scale_L8():
    print(f"\n{'=' * 78}")
    print("  EXPANSION 7: L=8, n_k=4 (N=256, 16 slices)")
    print(f"{'=' * 78}")
    C2, c1 = compute_second_chern(L=8, n_k=4, gamma_halt=0.0)
    nz = sum(1 for c in c1 if c != 0)
    print(f"  Gamma=0: C2={C2:+d}  C1 non-zero: {nz}/16")
    C2h, c1h = compute_second_chern(L=8, n_k=4, gamma_halt=15.0)
    nzh = sum(1 for c in c1h if c != 0)
    print(f"  Gamma=15: C2={C2h:+d}  C1 non-zero: {nzh}/16")
    print(f"  {'='*78}")


# ======================================================================
# 15. Main
# ======================================================================

if __name__ == "__main__":
    run_4d_oracle(L=4, n_k=4)
    gamma_annihilation_sweep()
    m0_sweep()
    scale_up()
    gamma_at_topological()
    m0_sweep_L6()
    uniform_gamma_annihilation()
    scale_L8()
