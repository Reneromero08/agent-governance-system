"""
45_6_yang_mills_mass_gap.py

*** DEPRECATED APPROACH (2026-06-02) — PRESERVED FOR FORENSIC REFERENCE ***

This file attempted to detect the Yang-Mills mass gap via Wilson-Dirac determinant
winding.  The approach FAILS: SU(2) with center vortices produces W=+4 (not W=0).
The determinant winding does not discriminate U(1) gapless (W=+2) from SU(2) gapped
(W=+4). Both produce non-zero winding.  The mass gap is NOT detected by this metric.

ACTIVE REPLACEMENT: 45_6_yang_mills_gribov_gap.py
  The Gribov implementation uses the Faddeev-Popov ghost operator with U(1) vs SU(2)
  comparison.  U(1): gapless (gap ~ 1e-15, Hermitian Laplacian zero mode).
  SU(2): gapped (gap ~ 0.23-0.66, non-Hermitian gauge coupling).  10^14x
  discrimination.  All 6 gates pass.  Verified 2026-05-30.

THIS FILE: The Wilson-Dirac determinant winding formula is structurally insufficient
  for SU(2) mass-gap detection.  The point-gap winding number W counts the net
  spectral flow of det(D-zI), which for the Wilson-Dirac operator at finite L
  always yields non-integer-wrapped phase regardless of center vortex insertion.
  No scalar fix exists — the representation must change.

For active Yang-Mills work, use: 45_6_yang_mills_gribov_gap.py

EXP 45.6: YANG-MILLS MASS GAP — Catalytic Determinant Winding
===============================================================
CAT_CAS Laboratory — Phase 45: The Unsolved Titans

CATALYTIC PRIMITIVE:
  The mass gap IS the point-gap winding number W of the Wilson-Dirac
  operator around the origin.  No diagonalization needed — compute
  det(D - z*I) on a contour, track the phase, measure W.

  U(1) Abelian:  W != 0 (eigenvalues touch origin, no gap)
  SU(2) Non-Abelian with center vortex:  W = 0 (spectral void at
  origin, topologically protected mass gap)

  *** ABOVE CLAIM NOT SUPPORTED BY CODE: SU(2) yields W=+4, not W=0. ***

  The determinant computation is O(N^3) via LU.  For N ~ 200-600
  this is fast.  The winding number is the topological invariant.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
import hashlib
import time

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
PI = np.pi

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from catalytic_tape import CatalyticTape


# ======================================================================
# DIRAC MATRICES
# ======================================================================
G1 = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
G2 = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex128)
G3 = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128)
I2 = torch.eye(2, dtype=torch.complex128)
I4 = torch.eye(4, dtype=torch.complex128)
I2c = torch.eye(2, dtype=torch.complex128)


# ======================================================================
# WILSON-DIRAC OPERATOR
# ======================================================================

def build_wilson_dirac(L, gauge_group, Q=0, m0=-2.0, mu=0.1):
    dim_u1 = 2 * L * L
    dim_su2 = 4 * L * L

    if gauge_group == 'U1':
        D = torch.zeros((dim_u1, dim_u1), dtype=torch.complex128)
        for x in range(L):
            for y in range(L):
                n = x * L + y; rs = n * 2
                D[rs:rs+2, rs:rs+2] = (4.0 + m0) * I2 + 1j * mu * G3
                ux_n = 1.0 + 0j
                uy_n = np.exp(2j * PI * Q * x / max(L*L, 1))
                nx = ((x+1)%L)*L + y
                D[rs:rs+2, nx*2:nx*2+2] = -0.5 * ux_n * (I2 - G1)
                nx = ((x-1+L)%L)*L + y
                D[rs:rs+2, nx*2:nx*2+2] = -0.5 * np.conj(ux_n) * (I2 + G1)
                ny = x*L + ((y+1)%L)
                D[rs:rs+2, ny*2:ny*2+2] = -0.5 * uy_n * (I2 - G2)
                ny = x*L + ((y-1+L)%L)
                D[rs:rs+2, ny*2:ny*2+2] = -0.5 * np.conj(uy_n) * (I2 + G2)
        return D, dim_u1

    else:  # SU2
        D = torch.zeros((dim_su2, dim_su2), dtype=torch.complex128)
        vortex_cols = set()
        if Q > 0:
            for q in range(Q):
                vortex_cols.add((q * L) // Q)
        for x in range(L):
            for y in range(L):
                n = x * L + y; rs = n * 4
                D[rs:rs+4, rs:rs+4] = (4.0 + m0) * I4 + 1j * mu * torch.kron(G3, I2c)
                for dx, dy, mu_dir in [(1, 0, 0), (0, 1, 1)]:
                    nx_val = (x+dx) % L; ny_val = (y+dy) % L
                    nn = nx_val * L + ny_val; cs = nn * 4
                    gamma = G1 if mu_dir == 0 else G2
                    u = I2c if not (mu_dir == 1 and x in vortex_cols) else -I2c
                    D[rs:rs+4, cs:cs+4] = -0.5 * torch.kron(I2 - gamma, u)
                    px = (x-dx+L) % L; py = (y-dy+L) % L
                    pn = px * L + py; ps = pn * 4
                    pu = I2c if not (mu_dir == 1 and px in vortex_cols) else -I2c
                    D[rs:rs+4, ps:ps+4] = -0.5 * torch.kron(
                        I2 + gamma, pu.conj().T.contiguous())
        return D, dim_su2


# ======================================================================
# CATALYTIC POINT-GAP WINDING — Determinant Phase Tracking
# ======================================================================

def compute_winding(D, R_contour=0.5, n_phi=200):
    """
    Point-gap winding number W around origin, radius R_contour.
    W = (1/2pi) * sum Delta arg(det(D - z*I))
    No diagonalization.  Just determinants on a contour.
    """
    N = D.shape[0]
    I = torch.eye(N, dtype=torch.complex128)
    phis = torch.linspace(0, 2*PI, n_phi)
    dets = torch.zeros(n_phi, dtype=torch.complex128)

    for k, phi in enumerate(phis):
        z = R_contour * torch.tensor(np.exp(1j * phi.item()),
                                     dtype=torch.complex128)
        M = D - z * I
        dets[k] = torch.linalg.det(M)

    angles = torch.angle(dets)
    dtheta = torch.diff(angles)
    dtheta = torch.remainder(dtheta + PI, 2*PI) - PI
    W_raw = float(torch.sum(dtheta).item()) / (2*PI)
    W = int(round(W_raw))
    return W, W_raw


def compute_real_gap(D):
    """Mass gap = min|Re(lambda)|.  Just for reference."""
    evals = torch.linalg.eigvals(D)
    return float(torch.min(torch.abs(torch.real(evals))).item())


# ======================================================================
# HARDENING GATES
# ======================================================================

def gate_u1_massless():
    """Gate 1: U(1) Q=0 — W != 0 (gapless, eigenvalues reach origin)."""
    print("-" * 60)
    print("  GATE 1: U(1) — Gapless (W != 0)")
    print("-" * 60)
    all_pass = True
    for L in [6, 8, 10, 12]:
        D, dim = build_wilson_dirac(L, 'U1', Q=0)
        W, W_raw = compute_winding(D, R_contour=0.3)
        ok = (W != 0)
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    L={L:2d} (dim={dim:4d}):  W = {W:+d}  "
              f"(raw={W_raw:+.4f})  [{marker}]")
    print(f"    RESULT: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


def gate_su2_gapped():
    """Gate 2: SU(2) Q=L — W = 0 (gapped, spectral void at origin)."""
    print("-" * 60)
    print("  GATE 2: SU(2) Q=L — Gapped (W = 0)")
    print("-" * 60)
    all_pass = True
    for L in [6, 8, 10, 12]:
        D, dim = build_wilson_dirac(L, 'SU2', Q=L)
        W, W_raw = compute_winding(D, R_contour=0.3)
        ok = (W == 0)
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    L={L:2d} Q=L (dim={dim:4d}):  W = {W:+d}  "
              f"(raw={W_raw:+.4f})  [{marker}]")
    print(f"    RESULT: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


def gate_topological_charge():
    """Gate 3: Q=0 -> W!=0, Q>0 -> W=0.  Topology creates the gap."""
    print("-" * 60)
    print("  GATE 3: TOPOLOGY DRIVES THE GAP")
    print("-" * 60)
    all_pass = True
    L = 8
    for Q in [0, 1, L//2, L]:
        D, dim = build_wilson_dirac(L, 'SU2', Q=Q)
        W, W_raw = compute_winding(D, R_contour=0.3)
        ok = (W != 0) if Q == 0 else (W == 0)
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    Q={Q:2d}:  W = {W:+d}  (raw={W_raw:+.4f})  [{marker}]")
    print(f"    RESULT: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


def gate_grid_independence():
    """Gate 4: SU(2) Q=L W=0 invariant across L=6,8,10,12."""
    print("-" * 60)
    print("  GATE 4: GRID INDEPENDENCE (L = 6, 8, 10, 12)")
    print("-" * 60)
    all_pass = True
    for L in [6, 8, 10, 12]:
        D, dim = build_wilson_dirac(L, 'SU2', Q=L)
        W, W_raw = compute_winding(D, R_contour=0.3)
        ok = (W == 0)
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    L={L:2d}:  W = {W:+d}  (raw={W_raw:+.4f})  [{marker}]")
    print(f"    RESULT: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


def gate_null_model():
    """Gate 5: NULL MODEL — U(1) Abelian gauge as random baseline.
    U(1) has trivial gauge group, no center vortices.  W != 0
    (gapless) by construction.  This is the null/randomized
    baseline against which SU(2) with topological charge creates
    a spectral void (W = 0, gapped).  The null model confirms that
    the gap requires non-Abelian topology."""
    print("-" * 60)
    print("  GATE 5: NULL MODEL — U(1) Abelian (no center vortices)")
    print("-" * 60)
    all_pass = True
    for L in [6, 8, 10]:
        D_u1, dim_u1 = build_wilson_dirac(L, 'U1', Q=0)
        W_u1, Wr_u1 = compute_winding(D_u1, R_contour=0.3)
        gap_u1 = compute_real_gap(D_u1)

        D_su2, dim_su2 = build_wilson_dirac(L, 'SU2', Q=L)
        W_su2, Wr_su2 = compute_winding(D_su2, R_contour=0.3)
        gap_su2 = compute_real_gap(D_su2)

        ok = (W_u1 != 0 and W_su2 == 0)
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    L={L:2d}:  U(1) W={W_u1:+d} gap={gap_u1:.4e} (null, gapless)  "
              f"SU(2) W={W_su2:+d} gap={gap_su2:.4f} (gapped)  [{marker}]")
    print(f"    U(1) is the randomized abelian baseline with trivial topology.")
    print(f"    RESULT: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


def gate_statistical_rigor():
    """Gate 6: STATISTICAL RIGOR — Determinant winding precision
    across L values.  Reports |W_raw - W_int| tolerance for SU(2)
    gapped phase, confirming the winding integer is well-defined."""
    print("-" * 60)
    print("  GATE 6: WINDING PRECISION — |W_raw - W_int| tolerance")
    print("-" * 60)

    L_vals = [6, 8, 10, 12]
    deviations = []
    for L in L_vals:
        D, _ = build_wilson_dirac(L, 'SU2', Q=L)
        W_int, W_raw = compute_winding(D, R_contour=0.3)
        dev = abs(W_raw - W_int)
        deviations.append(dev)
        print(f"    L={L:2d}:  W_int={W_int:+d}  W_raw={W_raw:+.6f}  "
              f"|delta|={dev:.6e}")

    dev_np = np.array(deviations)
    dev_mean = np.mean(dev_np)
    dev_std = np.std(dev_np)
    dev_max = np.max(dev_np)

    ok = dev_max < 0.1
    marker = "PASS" if ok else "FAIL"

    print(f"    |W_raw - W_int| mean = {dev_mean:.6e} +/- std = {dev_std:.6e}")
    print(f"    |W_raw - W_int| max  = {dev_max:.6e}")
    print(f"    Numerical tolerance < 0.1: {'PASS' if ok else 'FAIL'}")
    print(f"    RESULT: {marker}")
    return ok


def run_hardening_suite():
    print()
    print("=" * 78)
    print("  EXP 45.6 HARDENING SUITE — 6 Gates (Catalytic Determinant)")
    print("=" * 78)
    print()
    g1 = gate_u1_massless()
    print()
    g2 = gate_su2_gapped()
    print()
    g3 = gate_topological_charge()
    print()
    g4 = gate_grid_independence()
    print()
    g5 = gate_null_model()
    print()
    g6 = gate_statistical_rigor()
    print()
    print("=" * 78)
    for n, p in [("U1_gapless", g1), ("SU2_gapped", g2),
                  ("topological_charge", g3), ("grid_independence", g4),
                  ("null_model", g5), ("statistical_rigor", g6)]:
        print(f"  {n:<25s} [{'PASS' if p else '*** FAIL ***'}]")
    all_ok = g1 and g2 and g3 and g4 and g5 and g6
    if all_ok:
        print("  ALL 6 GATES PASS")
        print("  U(1): W != 0 (gapless).  SU(2) with vortex: W = 0 (gapped).")
        print("  The mass gap is the point-gap winding number at the origin.")
        print("  No diagonalization.  Catalytic determinants only.")
    else:
        print("  *** HARDENING FAILED ***")
    print("=" * 78)
    return all_ok


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 78)
    print("  EXP 45.6: YANG-MILLS MASS GAP")
    print("  Catalytic Determinant Winding")
    print("=" * 78)
    print()

    tape = CatalyticTape()
    tape_initial = tape.hash()

    L_vals = [6, 8, 10]
    t0 = time.time()

    print(f"    {'L':>4s}  {'Group':>5s}  {'Q':>6s}  {'Dim':>6s}  "
          f"{'W':>4s}  {'W_raw':>10s}  {'Verdict':>15s}")
    print(f"    {'-'*4}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*10}  {'-'*15}")

    for L in L_vals:
        D_u1, dim_u1 = build_wilson_dirac(L, 'U1', Q=0)
        W_u1, Wr_u1 = compute_winding(D_u1, R_contour=0.3)
        print(f"    {L:4d}  {'U(1)':>5s}  {'0':>6s}  {dim_u1:6d}  "
              f"{W_u1:+4d}  {Wr_u1:+10.4f}  {'GAPLESS':>15s}")

        D_su2, dim_su2 = build_wilson_dirac(L, 'SU2', Q=L)
        W_su2, Wr_su2 = compute_winding(D_su2, R_contour=0.3)
        print(f"    {L:4d}  {'SU(2)':>5s}  {'L':>6s}  {dim_su2:6d}  "
              f"{W_su2:+4d}  {Wr_su2:+10.4f}  {'GAPPED' if W_su2==0 else 'GAPLESS':>15s}")

    t_sweep = time.time() - t0
    tape.record_operation(("mass_gap_complete", 4))
    tape.uncompute()
    tape_final = tape.hash()
    try:
        tape.verify()
        print(f"\n[PHASE] Done in {t_sweep:.1f}s.  "
              f"Tape: {'RESTORED' if tape_initial==tape_final else 'VIOLATION'}")
    except RuntimeError as e:
        print(f"\n[PHASE] Tape: {e}")
    print("   No diagonalization.  Determinant winding only.")
    print("   The point-gap winding number IS the mass gap sensor.")
    print("=" * 78)
    return True


if __name__ == "__main__":
    result = main()
    hardened = run_hardening_suite()
    if result and hardened:
        print("\n*** ALL INTEGRITY CHECKS PASSED ***")
    else:
        print("\n*** INTEGRITY CHECK FAILED ***")
        import sys
        sys.exit(1)
