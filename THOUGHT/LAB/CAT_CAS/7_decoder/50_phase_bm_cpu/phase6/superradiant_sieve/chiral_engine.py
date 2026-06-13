"""
chiral_engine.py - the non-Hermitian COLLECTIVE (superradiant) engine for Exp 50.14, in a
CHIRAL (handed) geometry. This is the lab's own validated radiative Hamiltonian
(H = Omega - i Upsilon/2) re-expressed as a 1D chiral-waveguide collective decay matrix.

WHAT THIS MODULE IS
-------------------
1. A faithful single-excitation collective non-Hermitian Hamiltonian for M emitters:
       H_eff[m,n] = (E_m - i gamma/2) delta_mn + (1 - delta_mn) V_mn
   with the standard Spano-Mukamel dipole coupling V_mn = Delta_mn - i Upsilon_mn / 2.
   VALIDATED by the Dicke test (2 parallel near-field dipoles -> Gamma/gamma = [2, 0]) and
   the sum rule (sum_j Gamma_j = M gamma), reproducing the lab superradiance result.

2. A CHIRAL (cascaded / directional waveguide) collective decay operator. For M emitters at
   1D positions z_j coupled to a channel with wavevector k0 and right/left decay rates
   gamma_R, gamma_L (chirality D = (gamma_R - gamma_L)/(gamma_R + gamma_L)):

       H_eff[m,n] = -i/2 gamma_R exp(i k0 (z_m - z_n))   if z_m > z_n
                  = -i/2 gamma_L exp(i k0 (z_n - z_m))   if z_m < z_n
                  = -i/2 (gamma_R + gamma_L)             if m == n

   The Hermitian collective DECAY matrix is Gamma_G = i (H_eff - H_eff^dagger). Two limits:
     - D = 0 (gamma_L = gamma_R, bidirectional): Gamma_G[m,n] = gamma cos(k0 |z_m - z_n|),
       REAL SYMMETRIC -> Im(Gamma_G) = 0. This array is MIRROR-SYMMETRIC (achiral): it
       COMMUTES WITH THE REFLECTION, so by the 50.14 black-hole result it is ORIENTATION-BLIND.
     - D = 1 (gamma_L = 0, cascaded): Gamma_G[m,n] = gamma exp(i k0 (z_m - z_n)) off-diagonal,
       with a NONZERO antisymmetric imaginary part Im(Gamma_G)[m,n] = gamma sin(k0 (z_m - z_n)).
       This array is CHIRAL: a FIXED, d-INDEPENDENT operator that does NOT commute with the
       mirror. It is the one loophole in the 50.14 wall ("any fixed operator commuting with R
       is blind"): a physical handedness reference.

   The chirality is the HANDEDNESS REFERENCE the orientation bit (a chirality of the coset
   phase, omega^{kd} vs omega^{-kd}) can ring against.

NO-SMUGGLE: the array geometry (z_j, k0, D) is FIXED and d-INDEPENDENT. The secret d enters
ONLY through the coset-state dipole phases carried by the received emitters (coset_array.py),
exactly as in the black-hole model (the OPERATOR is fixed; the STATE carries d).

ASCII only. RNGs seeded by caller. CPU dense diagonalization (M up to a few thousand).
Author: superradiant-sieve agent (Fable). Claim ceiling L4-5.
"""
import numpy as np
from scipy.linalg import eigvals, eigh

# ----- physical constants (lab superradiance values, cm^-1 / nm) -----
GAMMA = 0.00273          # single-emitter radiative decay rate (cm^-1)
LAMBDA_NM = 280.0        # UV transition wavelength (nm)
K_OPT = 2.0 * np.pi / LAMBDA_NM   # optical wavevector (nm^-1)


# ===========================================================================
# PART 1 - the VALIDATED dipole-dipole engine (Dicke test + sum rule)
# ===========================================================================
def dipole_coupling(r, gamma=GAMMA, k=K_OPT):
    """Complex parallel-dipole coupling V(r) = Delta(r) - i Gamma(r)/2 (Spano-Mukamel 1989).
    Reproduces the lab superradiance Hamiltonian. r in nm."""
    kr = k * r
    if kr < 1e-8:
        return complex(-3 * gamma / (4 * kr ** 3), -3 * gamma / 4)
    s, c = np.sin(kr), np.cos(kr)
    delta = (3 * gamma / 4) * (c / kr - s / kr ** 2 - c / kr ** 3)
    gamma_c = (3 * gamma / 2) * (s / kr + c / kr ** 2 - s / kr ** 3)
    return complex(delta, -gamma_c / 2)


def dicke_two_dipole(spacing_nm=0.05):
    """The canonical validation: two parallel near-field dipoles. The symmetric and
    antisymmetric collective modes have decay rates Gamma/gamma -> [2, 0] (super/subradiant).
    Returns the sorted decay-rate ratios."""
    H = np.zeros((2, 2), dtype=complex)
    H[0, 0] = H[1, 1] = complex(0.0, -GAMMA / 2)
    V = dipole_coupling(spacing_nm)
    H[0, 1] = H[1, 0] = V
    ev = eigvals(H)
    gj = np.sort(-2.0 * np.imag(ev))[::-1]
    return gj / GAMMA


def sum_rule_residual(M=40, spacing_nm=0.4, seed=0):
    """sum_j Gamma_j = M gamma (the radiative sum rule), to machine precision: trace of the
    anti-Hermitian part is conserved under diagonalization. Returns |sum Gamma_j - M gamma|/(M gamma)."""
    rng = np.random.default_rng(seed)
    H = np.zeros((M, M), dtype=complex)
    z = np.arange(M) * spacing_nm
    for m in range(M):
        H[m, m] = complex(rng.uniform(-100, 100) * 0, -GAMMA / 2)
    for m in range(M):
        for n in range(m + 1, M):
            V = dipole_coupling(abs(z[m] - z[n]))
            H[m, n] = H[n, m] = V
    ev = eigvals(H)
    gj = -2.0 * np.imag(ev)
    return abs(np.sum(gj) - M * GAMMA) / (M * GAMMA), gj / GAMMA


# ===========================================================================
# PART 2 - the CHIRAL collective decay operator (the handedness reference)
# ===========================================================================
def chiral_waveguide_H(z, k0, D=1.0, gamma=1.0):
    """Effective non-Hermitian Hamiltonian H_eff for M emitters at 1D positions z coupled to
    a chiral channel. D in [0,1] is the chirality: D=0 bidirectional (achiral, mirror-symmetric),
    D=1 cascaded (fully chiral). k0 is the channel wavevector. gamma sets the rate scale.

    gamma_R = gamma (1 + D)/2,  gamma_L = gamma (1 - D)/2.
    H[m,n] = -i/2 gamma_R e^{i k0 (z_m - z_n)} (z_m > z_n);  -i/2 gamma_L e^{i k0 (z_n - z_m)}
             (z_m < z_n);  -i/2 (gamma_R + gamma_L) (diagonal).
    FIXED operator, d-INDEPENDENT (no reference to the secret)."""
    M = len(z)
    gR = gamma * (1.0 + D) / 2.0
    gL = gamma * (1.0 - D) / 2.0
    H = np.zeros((M, M), dtype=complex)
    for m in range(M):
        H[m, m] = -0.5j * (gR + gL)
    for m in range(M):
        for n in range(M):
            if m == n:
                continue
            dz = z[m] - z[n]
            if dz > 0:
                H[m, n] = -0.5j * gR * np.exp(1j * k0 * dz)
            else:
                H[m, n] = -0.5j * gL * np.exp(1j * k0 * (-dz))
    return H


def traveling_wave_decay_matrix(z, k0, chiral=True, gamma=1.0):
    """The canonical chiral-quantum-optics collective decay matrix, in closed rank form.

      CHIRAL  (single directional / traveling-wave channel):
          Gamma_G = gamma w w^dagger,   w_m = exp(i k0 z_m)   (rank-1, PSD: ONE bright mode).
          Im(Gamma_G)_{mn} = gamma sin(k0 (z_m - z_n))  -> NONZERO antisymmetric handedness.
          Does NOT commute with the reflection: a fixed, d-independent handedness reference.

      ACHIRAL (balanced bidirectional / standing-wave channel):
          Gamma_G = gamma Re(w w^dagger) = gamma cos(k0 (z_m - z_n))  (rank-2, PSD).
          Im(Gamma_G) = 0  -> mirror-symmetric, COMMUTES WITH R: orientation-blind by the
          50.14 black-hole result.

    This is the cleanest faithful model (Dicke superradiance into one vs two counter-propagating
    channels) and is exactly the analytic form used by the O(M) fast statistic."""
    w = np.exp(1j * k0 * np.asarray(z, dtype=float))
    G = gamma * np.outer(w, np.conj(w))
    if chiral:
        return G
    return G.real.astype(complex)


def decay_matrix(H):
    """Hermitian collective decay matrix Gamma_G = i (H - H^dagger). PSD for a valid
    dissipator. Its IMAGINARY antisymmetric part is the chiral / handedness content."""
    return 1j * (H - H.conj().T)


def chiral_content(Gamma_G):
    """Returns (||Im(Gamma_G)||_F, parity_noncommute). Im(Gamma_G) is the antisymmetric
    handedness kernel; nonzero <=> chiral. parity_noncommute = ||[Gamma_G, P]||_F for the
    reflection P (reverse emitter order): 0 for achiral (commutes with the mirror), >0 chiral."""
    M = Gamma_G.shape[0]
    P = np.eye(M)[::-1].copy()
    imnorm = float(np.linalg.norm(Gamma_G.imag, ord="fro"))
    par = float(np.linalg.norm(Gamma_G @ P - P @ Gamma_G, ord="fro"))
    return imnorm, par


def bright_dark_modes(Gamma_G):
    """Diagonalize the Hermitian decay matrix: eigenvalues are the collective decay rates
    Gamma_mode (bright = large, dark = ~0). Returns (rates_desc, vecs) sorted descending."""
    w, V = eigh(Gamma_G)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


if __name__ == "__main__":
    gj = dicke_two_dipole()
    res, _ = sum_rule_residual()
    z = np.arange(8, dtype=float)
    for D, lab in [(0.0, "achiral"), (1.0, "chiral")]:
        H = chiral_waveguide_H(z, k0=0.7, D=D)
        G = decay_matrix(H)
        imn, par = chiral_content(G)
        rates, _ = bright_dark_modes(G)
        print("%-8s  ||Im(Gamma)||=%.3e  [Gamma,P]=%.3e  bright/dark rates=%s" %
              (lab, imn, par, np.round(rates, 3)))
    print("Dicke 2-dipole Gamma/gamma =", np.round(gj, 4), " (expect [2,0])")
    print("sum-rule residual =", res)
