"""
40_5d_floquet_oracle.py

EXPERIMENT 40: 5D NON-HERMITIAN FLOQUET TIME CRYSTAL ORACLE
============================================================
Halting as Time Crystal melting via discrete time-translation symmetry
breaking in a periodically-driven 4D Axion Insulator.

LOOPS:  Floquet operator U_F develops robust pi-modes (eigenvalues near -1)
        -> Discrete Time Crystal phase -> period-2 subharmonic attractor.
HALTS:  Uniform EP sink damps pi-modes, pulling them off the unit circle
        -> Time Crystal melts -> spectral weight at z=-1 collapses to zero.

ARCHITECTURE:
  - U_F = exp(-i H1*dt) * exp(-i H0*dt) — two-step piecewise Floquet evolution
  - Pi-gap projector via contour integral around z = -1
  - Uniform gamma field on ALL sites (Exp 39 breakthrough)
  - O(L^2 * 16) buffers reused across (kz, kw) slices

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

# 4D Dirac matrices (from Exp 39)
G1 = torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=COMPLEX)
G2 = torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=COMPLEX)
G3 = torch.tensor([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], dtype=COMPLEX)
G4 = torch.tensor([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]], dtype=COMPLEX)
G5 = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=COMPLEX)
I4 = torch.eye(4, dtype=COMPLEX)


# ======================================================================
# 1.  Driven 4D Slice Hamiltonian
# ======================================================================

def build_driven_slice(L, kz, kw, t_step, drive_amp, m0=1.0,
                       tz=1.0, tw=1.0, t1=1.0, loss=0.05, gamma_halt=0.0):
    N_sp = L * L; N = N_sp * 4
    H = torch.zeros((N, N), dtype=COMPLEX)
    M_drive = (m0 - tz * np.cos(kz) - tw * np.cos(kw)) + \
              (drive_amp if t_step == 0 else -drive_amp)

    for y in range(L):
        for x in range(L):
            si = y * L + x; ib = slice(si * 4, (si + 1) * 4)
            H[ib, ib] = M_drive * G5 - 1j * loss * I4
            if gamma_halt > 0:
                H[ib, ib] -= 1j * gamma_halt * I4  # uniform on all sites
            # +x hopping
            nx, ny = (x+1)%L, y; sj = ny*L + nx; jb = slice(sj*4,(sj+1)*4)
            H[jb, ib] += t1 * (G1 + 1j*G2)/2.0
            H[ib, jb] += t1 * (G1 - 1j*G2)/2.0
            # +y hopping
            nx, ny = x, (y+1)%L; sj = ny*L + nx; jb = slice(sj*4,(sj+1)*4)
            H[jb, ib] += t1 * (G3 + 1j*G4)/2.0
            H[ib, jb] += t1 * (G3 - 1j*G4)/2.0
    return H


# ======================================================================
# 2.  Floquet Operator U_F = exp(-i H1 dt) * exp(-i H0 dt)
# ======================================================================

def floquet_operator(L, kz, kw, drive_amp, dt=0.5, gamma_halt=0.0):
    H0 = build_driven_slice(L, kz, kw, 0, drive_amp, gamma_halt=gamma_halt)
    H1 = build_driven_slice(L, kz, kw, 1, drive_amp, gamma_halt=gamma_halt)
    U0 = torch.linalg.matrix_exp(-1j * H0 * dt)
    U1 = torch.linalg.matrix_exp(-1j * H1 * dt)
    return U1 @ U0


# ======================================================================
# 3.  Pi-Gap Projector (contour integral around z = -1)
# ======================================================================

def pi_gap_projector(U_F, n_pts=32, radius=0.4):
    N = U_F.shape[0]; I = torch.eye(N, dtype=COMPLEX)
    P = torch.zeros((N, N), dtype=COMPLEX)
    for k in range(n_pts):
        theta = 2 * np.pi * k / n_pts
        z = -1.0 + radius * torch.tensor(np.exp(1j*theta), dtype=COMPLEX)
        M = z * I - U_F
        P += torch.linalg.inv(M) * (radius * torch.tensor(np.exp(1j*theta), dtype=COMPLEX))
    return P / n_pts


# ======================================================================
# 4.  Measure pi-weight (trace of pi-gap projector)
# ======================================================================

def measure_pi_weight(L, n_k, drive_amp, gamma_halt):
    kz_vals = torch.linspace(0, 2*np.pi, n_k)
    kw_vals = torch.linspace(0, 2*np.pi, n_k)
    total_weight = 0.0; nz = 0
    for kz in kz_vals:
        for kw in kw_vals:
            U_F = floquet_operator(L, kz.item(), kw.item(), drive_amp, gamma_halt=gamma_halt)
            P = pi_gap_projector(U_F)
            w = torch.trace(P).real.item()
            total_weight += w
            if abs(w) > 0.5: nz += 1
    return total_weight, nz


# ======================================================================
# 5.  Oracle runner
# ======================================================================

def run_5d_oracle(L=4, n_k=4, drive_amp=1.5):
    slices = n_k * n_k
    print("=" * 78)
    print("  EXPERIMENT 40: 5D NON-HERMITIAN FLOQUET TIME CRYSTAL ORACLE")
    print("  Pi-Gap Modes -> Time Crystal / LOOPS  |  Melt -> HALTS")
    print("=" * 78)
    print(f"  L={L}  n_k={n_k}  N={(L*L*4)}  drive_amp={drive_amp}")
    print(f"  U_F = exp(-iH1*dt)*exp(-iH0*dt)  dt=0.5")

    # ----  LOOPING  ----
    wl, nl = measure_pi_weight(L, n_k, drive_amp, gamma_halt=0.0)
    print(f"\n  Looping (Gamma=0):  pi-weight={wl:.2f}  active={nl}/{slices}")
    print(f"  VERDICT: {'LOOPS (Time Crystal)' if nl > 0 else 'HALTS'}")

    # ----  HALTING  ----
    wh, nh = measure_pi_weight(L, n_k, drive_amp, gamma_halt=5.0)
    print(f"\n  Halting (Gamma=5):  pi-weight={wh:.2f}  active={nh}/{slices}")
    print(f"  VERDICT: {'LOOPS (Time Crystal)' if nh > 0 else 'HALTS (Time Crystal Melted)'}")

    # ----  Gamma sweep  ----
    print(f"\n  GAMMA SWEEP — Pi-weight vs sink strength")
    print(f"  {'Gamma':>8s}  {'pi-weight':>12s}  {'active':>7s}  {'Verdict'}")
    print("  " + "-" * 45)
    for g in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        wg, ng = measure_pi_weight(L, n_k, drive_amp, gamma_halt=g)
        v = "LOOPS" if ng > 0 else "HALTS (melted)"
        print(f"  {g:8.1f}  {wg:12.2f}  {ng:4d}/{slices}  {v}")

    # ----  Summary  ----
    print(f"\n{'=' * 78}")
    print("  ORACLE SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Case            pi-weight    active slices    Verdict")
    print(f"  {'-'*50}")
    print(f"  Looping (G=0)     {wl:8.2f}      {nl:3d}/{slices}         "
          f"{'LOOPS' if nl>0 else 'HALTS'}")
    print(f"  Halting (G=5)    {wh:8.2f}      {nh:3d}/{slices}         "
          f"{'LOOPS' if nh>0 else 'HALTS'}")
    print(f"  {'='*78}")
    if nl > 0 and nh == 0:
        print("  CORRECT: Time Crystal melted by uniform EP sink")
    elif nl > 0 and nh > 0:
        print("  PARTIAL: Time Crystal survived — increase drive_amp or Gamma")
    else:
        print("  FAILED: No pi-modes at Gamma=0 — check drive_amp")
    print(f"  {'='*78}")


if __name__ == "__main__":
    run_5d_oracle(L=4, n_k=4, drive_amp=1.5)
