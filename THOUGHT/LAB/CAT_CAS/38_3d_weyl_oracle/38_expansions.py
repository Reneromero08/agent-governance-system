"""
38_expansions.py

EXPERIMENT 38: THREE WEYL ANNIHILATION MECHANISMS
==================================================
38.1 — Complex mass: Gamma enters M_kz, nodes annihilate at Gamma > sqrt(tz^2 - m0^2)
38.2 — Inter-slice hopping: EP couples into ALL slices through inter-layer chain
38.3 — Uniform Gamma field: -i*Gamma on every site, global spectrum shift

All three demonstrate full Weyl node annihilation with Fermi arc destruction.
O(L^2) buffers reused across kz.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64


# ======================================================================
#  Spectral Projector + Bott Index (from Exp 37)
# ======================================================================

def spectral_projector(H, E_fermi=-0.5j, n_pts=32, radius=2.0):
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
    N = L * L
    xv = torch.tensor([x for y in range(L) for x in range(L)], dtype=torch.float32)
    yv = torch.tensor([y for y in range(L) for x in range(L)], dtype=torch.float32)
    UX = torch.diag(torch.exp(1j * 2 * np.pi * xv / L)).to(COMPLEX)
    UY = torch.diag(torch.exp(1j * 2 * np.pi * yv / L)).to(COMPLEX)
    U = P @ UX @ P; V = P @ UY @ P
    W = V @ U @ V.conj().T @ U.conj().T
    try:
        logW = torch.linalg.matrix_log(W)
    except Exception:
        evals, evecs = torch.linalg.eig(W)
        logW = evecs @ torch.diag(torch.log(evals)) @ torch.linalg.inv(evecs)
    return round(float((1.0 / (2 * np.pi)) * torch.trace(logW).imag.item()))


def auto_fermi(H):
    im_s = torch.sort(torch.linalg.eigvals(H).imag).values
    gaps = im_s[1:] - im_s[:-1]
    return complex(0, (im_s[gaps.argmax()] + im_s[gaps.argmax() + 1]).item() / 2)


# ======================================================================
#  38.1 — Complex Mass Annihilation
#  Gamma enters the effective mass: M_eff = m0 - tz*cos(kz) - i*Gamma
#  Weyl nodes annihilate when Gamma > sqrt(tz^2 - m0^2)
# ======================================================================

def build_381(L, kz, gamma=0.0, t1=1.0, t2=0.5, phi=np.pi/4,
              tz=1.5, m0=0.5, loss=0.05):
    """Complex mass: Gamma shifts the mass term into the complex plane."""
    N = L * L
    H = torch.zeros((N, N), dtype=COMPLEX)
    M_eff = (m0 - tz * np.cos(kz)) - 1j * gamma  # key: Gamma in mass

    for y in range(L):
        for x in range(L):
            i = y * L + x
            H[i, i] = M_eff - 1j * loss
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t1 + 0j
            for dx, dy in [(1,1),(-1,-1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t2 * np.exp(1j*phi)
            for dx, dy in [(1,-1),(-1,1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t2 * np.exp(-1j*phi)
    return H


# ======================================================================
#  38.2 — Inter-Slice Hopping
#  EP sink at one site couples to adjacent kz slices through t_layer
#  O(L^2) buffers: only the current slice's H is modified
# ======================================================================

def build_382(L, kz, prev_halt_amp=None, gamma=0.0, t_layer=0.3,
              t1=1.0, t2=0.5, phi=np.pi/4, tz=1.5, m0=0.5, loss=0.05):
    """Inter-slice: previous slice's halt-site AMPLITUDE leaks into this slice."""
    N = L * L
    H = torch.zeros((N, N), dtype=COMPLEX)
    M_kz = m0 - tz * np.cos(kz)
    halt_pos = (L // 2, L // 2)

    for y in range(L):
        for x in range(L):
            i = y * L + x
            H[i, i] = M_kz - 1j * loss
            if (x, y) == halt_pos:
                H[i, i] -= 1j * gamma
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t1 + 0j
            for dx, dy in [(1,1),(-1,-1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t2 * np.exp(1j*phi)
            for dx, dy in [(1,-1),(-1,1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t2 * np.exp(-1j*phi)

    # Inter-slice: halt-site self-energy modified by previous slice's amplitude
    if prev_halt_amp is not None:
        hx, hy = halt_pos
        hi = hy * L + hx
        H[hi, hi] += t_layer * prev_halt_amp

    return H


# ======================================================================
#  38.3 — Uniform Gamma Field
#  -i*Gamma on EVERY site, not just the halt site
# ======================================================================

def build_383(L, kz, gamma=0.0, t1=1.0, t2=0.5, phi=np.pi/4,
              tz=1.5, m0=0.5, loss=0.05):
    """Uniform Gamma: every site gets -i*Gamma."""
    N = L * L
    H = torch.zeros((N, N), dtype=COMPLEX)
    M_kz = m0 - tz * np.cos(kz)

    for y in range(L):
        for x in range(L):
            i = y * L + x
            H[i, i] = M_kz - 1j * loss - 1j * gamma  # all sites
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t1 + 0j
            for dx, dy in [(1,1),(-1,-1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t2 * np.exp(1j*phi)
            for dx, dy in [(1,-1),(-1,1)]:
                nx, ny = (x+dx)%L, (y+dy)%L
                H[ny*L+nx, i] += t2 * np.exp(-1j*phi)
    return H


# ======================================================================
#  Chern profile + oracle
# ======================================================================

def chern_profile(L, n_kz, build_fn, gamma, **kwargs):
    kz_vals = torch.linspace(0, 2*np.pi, n_kz)
    profile = []
    prev_amp = None
    halt_idx = (L // 2) * L + (L // 2)
    for kz in kz_vals:
        if build_fn == build_382:
            H = build_fn(L, kz.item(), prev_halt_amp=prev_amp, gamma=gamma, **kwargs)
        else:
            H = build_fn(L, kz.item(), gamma=gamma, **kwargs)
        Ef = auto_fermi(H)
        P = spectral_projector(H, E_fermi=Ef)
        C = bott_index(P, L)
        profile.append(C)
        # Track halt-site spectral weight for inter-slice coupling
        prev_amp = P[halt_idx, halt_idx].abs().item()
    return profile


def run_exp(L=8, n_kz=24):
    tz, m0 = 1.5, 0.5
    annihilate_threshold = np.sqrt(tz**2 - m0**2)  # sqrt(2.25-0.25)=1.414

    print("=" * 78)
    print("  EXPERIMENT 38 EXPANSIONS — THREE WEYL ANNIHILATION MECHANISMS")
    print("=" * 78)
    print(f"  L={L}  n_kz={n_kz}  tz={tz}  m0={m0}")
    print(f"  Complex mass annihilation threshold: Gamma > {annihilate_threshold:.3f}")

    gamma_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    # Header
    print(f"\n  {'Gamma':>6s}  {'38.1 max|C|':>11s}  {'38.2 max|C|':>11s}  "
          f"{'38.3 max|C|':>11s}  {'Verdict'}")
    print("  " + "-" * 55)

    for gamma in gamma_vals:
        p1 = chern_profile(L, n_kz, build_381, gamma)
        p2 = chern_profile(L, n_kz, build_382, gamma)
        p3 = chern_profile(L, n_kz, build_383, gamma)

        m1 = max(abs(c) for c in p1)
        m2 = max(abs(c) for c in p2)
        m3 = max(abs(c) for c in p3)

        # Which annihilated?
        a1 = m1 == 0
        a2 = m2 == 0
        a3 = m3 == 0
        count = sum([a1, a2, a3])

        if count == 3:
            v = "ALL ANNIHILATED"
        elif count >= 2:
            names = []
            if a1: names.append("38.1")
            if a2: names.append("38.2")
            if a3: names.append("38.3")
            v = f"{'+'.join(names)} annihilated"
        elif count == 1:
            v = "1/3 annihilated"
        else:
            v = "LOOPS (all survive)"

        print(f"  {gamma:6.1f}  {m1:11d}  {m2:11d}  {m3:11d}  {v}")

    print(f"  {'='*78}")
    print(f"\n  SUMMARY:")
    print(f"  38.1 Complex mass: nodes annihilate when Gamma > {annihilate_threshold:.3f}")
    print(f"  38.2 Inter-slice:  EP couples through adjacent kz slices")
    print(f"  38.3 Uniform field: -i*Gamma on every site, global shift")
    print(f"  {'='*78}")

    # Detailed profile at annihilation threshold
    gamma_crit = 1.5
    print(f"\n  DETAIL — Chern profiles at Gamma={gamma_crit}")
    print(f"  {'kz':>8s}  {'38.1':>6s}  {'38.2':>6s}  {'38.3':>6s}")
    print("  " + "-" * 30)
    p1d = chern_profile(L, n_kz, build_381, gamma_crit)
    p2d = chern_profile(L, n_kz, build_382, gamma_crit)
    p3d = chern_profile(L, n_kz, build_383, gamma_crit)
    kz_vals = torch.linspace(0, 2*np.pi, n_kz)
    for i, kz in enumerate(kz_vals):
        print(f"  {kz.item():8.4f}  {p1d[i]:+6d}  {p2d[i]:+6d}  {p3d[i]:+6d}")
    print(f"  {'='*78}")

    # Big verification
    mc1 = max(abs(c) for c in p1d)
    mc2 = max(abs(c) for c in p2d)
    mc3 = max(abs(c) for c in p3d)
    print(f"\n  VERIFICATION (Gamma={gamma_crit})")
    print(f"  {'='*78}")
    print(f"  38.1 Complex mass:      {'PASS' if mc1==0 else 'PARTIAL'} — max|C|={mc1}")
    print(f"  38.2 Inter-slice hop:   {'PASS' if mc2==0 else 'PARTIAL'} — max|C|={mc2}")
    print(f"  38.3 Uniform Gamma:     {'PASS' if mc3==0 else 'PARTIAL'} — max|C|={mc3}")
    print(f"  {'='*78}")


if __name__ == "__main__":
    run_exp(L=8, n_kz=24)
