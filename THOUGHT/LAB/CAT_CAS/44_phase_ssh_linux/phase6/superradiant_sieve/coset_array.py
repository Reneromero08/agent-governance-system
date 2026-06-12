"""
coset_array.py - encode the 50.14 coherent COSET STATES as emitters in the chiral
superradiant array, and read the ORIENTATION out of the collective chiral observable.

THE ENCODING (no-smuggle)
-------------------------
The oracle hands us M single-qubit coset states |c_{k_j}> = (|0> + omega^{k_j d}|1>)/sqrt2
for random KNOWN labels k_j (omega = exp(-2 pi i / N), N = 2^n). Emitter j IS this coset
state: its transition-dipole phase is p_j = omega^{k_j d} (a UNIT complex number; the
secret d lives in the phase, exactly as in the black-hole model). The labels k_j are public;
we never look at d.

The fixed, d-INDEPENDENT chiral array (chiral_engine) provides the collective decay operator
Gamma_G (Hermitian). For phased dipoles mu_j = p_j the physical collective emission rate of
the array is the quadratic form

      R(d) = v(d)^dagger Gamma_G v(d),     v_j(d) = p_j / sqrt(M).

KEY IDENTITY (the wall, and the loophole):
  Writing D(d) = diag(p_j) (unitary), the PHASED Hamiltonian is H(d) = D^dagger H_geom D, a
  SIMILARITY transform of the fixed geometry. Hence the collective decay RATES (eigenvalues /
  the bright-mode decay rate) are d-INVARIANT -- the orientation does NOT ring in any
  eigenvalue (the array-level restatement of the 50.14 B3 result: a fixed operator's spectrum
  is fold-blind). What CAN ring is the chiral EMISSION asymmetry of the specific phased input:

      R(d) - R(N-d) = v^dagger (Gamma_G - Gamma_G^T) v = 2i v^dagger Im(Gamma_G) v,

  the pure handedness statistic
      T(d) = (1/i) v(d)^dagger [ (Gamma_G - Gamma_G^T)/2 ] v(d)
           = -(2/M) sum_{j<l} Im(Gamma_G)_{jl} sin(2 pi (k_l - k_j) d / N).
  ACHIRAL array: Im(Gamma_G) = 0  =>  T == 0  =>  AUC = 0.5 EXACTLY (mirror-blind control).
  CHIRAL array:  Im(Gamma_G) != 0 =>  T odd in d (T(N-d) = -T(d))  =>  sign(T) = orientation.

THE SCALING (the deliverable). With z_j = k_j the chiral kernel resonates as
sin(k0 (k_l - k_j)) sin(2 pi d (k_l - k_j)/N): a FIXED k0 cannot match the unknown 2 pi d / N,
so for random labels the pair-sum off-resonance fluctuates (exp cost, like B1). The DYADIC
ladder concentrates the differences on powers of two -- whether that buys a poly / subexp drop
in the array size M(n) is what we measure. We model one collective SHOT honestly: M emitters
emit ~M/2 photons, the chiral asymmetry a(d) = T(d)/R_tot(d) is read with shot std ~
1/sqrt(M/2); superradiant enhancement and shot noise both scale with M, so M(n) is the
required array size to resolve sign(a) -- directly comparable to B1 (Theta(N)) and Kuperberg
(2^{O(sqrt n)}).

ASCII only. RNGs seeded by caller. Reuses construction.py + chiral_engine.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(HERE)
FOLD = os.path.join(PHASE6, "fold_audit")
for _p in (FOLD, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C
import chiral_engine as E


# ===========================================================================
# coset phases and the collective phased input
# ===========================================================================
def coset_phases(k, d, N):
    """p_j = omega^{k_j d} = exp(-2 pi i k_j d / N) (unit complex dipole phase of emitter j)."""
    return np.exp(-2j * np.pi * (k.astype(np.int64) * int(d) % N) / N)


def phased_input(k, d, N):
    """v(d) = (1/sqrt M) sum_j p_j |j>, the collective state of the M phased coset emitters."""
    p = coset_phases(k, d, N)
    return p / np.sqrt(len(p))


# ===========================================================================
# frequency sets (labels) and the fixed chiral array for a label set
# ===========================================================================
def freq_set(kind, M, n, rng):
    """Return M public labels k_j in [0, N).
      'dyadic'  : the ladder {1,2,4,...,N/2} (n rungs) replicated to length M (extended dyadic).
      'random'  : uniform random labels.
      'matched' : an arithmetic comb k_j = j+1 (small contiguous frequencies) - a fixed set
                  chosen to resonate with a small fixed helix wavevector (d-independent)."""
    N = 1 << n
    if kind == "dyadic":
        rungs = np.array([1 << j for j in range(n)], dtype=np.int64)  # 1,2,...,N/2
        reps = int(np.ceil(M / len(rungs)))
        k = np.tile(rungs, reps)[:M]
        return k.astype(np.int64)
    if kind == "matched":
        k = (1 + np.arange(M)) % N
        k[k == 0] = 1
        return k.astype(np.int64)
    # random
    return rng.integers(1, N, size=M).astype(np.int64)


def array_decay_matrix(k, n, k0, D, position="frequency"):
    """Build the FIXED collective decay matrix Gamma_G for the emitter label set k.
    position='frequency' -> z_j = k_j (the chiral coupling resonance is in label space);
    position='uniform'   -> z_j = rank(k_j) (a uniform helix, emitters ordered by label).
    D selects the channel: D>=1 CHIRAL (traveling wave), D==0 ACHIRAL (standing wave). k0, D
    are FIXED, d-INDEPENDENT array knobs. (The cascaded-waveguide engine variant lives in
    chiral_engine.chiral_waveguide_H; this canonical rank form is exact and faster.)"""
    if position == "frequency":
        z = k.astype(float)
    else:
        z = np.argsort(np.argsort(k)).astype(float)  # rank order along the helix
    return E.traveling_wave_decay_matrix(z, k0=k0, chiral=(D != 0.0), gamma=1.0)


# ===========================================================================
# the orientation readouts: full collective rate R, pure chiral statistic T, asymmetry a
# ===========================================================================
def readouts(Gamma_G, k, d, N):
    """Return (R, T, a):
      R = v^dagger Gamma_G v          (full collective emission rate, real)
      T = chiral handedness statistic (odd in d; sign = orientation for a chiral array)
      a = T / R                       (normalized chiral asymmetry in ~[-1,1])

    The orientation-odd part of the real rate R is  R(d) - R(N-d) = -2 T(d) with
      T(d) = Im( v^dagger Im(Gamma_G) v ),
    the antisymmetric (handedness) form of the real antisymmetric kernel B = Im(Gamma_G).
    For an ACHIRAL array B = 0 so T == 0 (mirror-blind); for a CHIRAL array T is odd in d."""
    v = phased_input(k, d, N)
    Gv = Gamma_G @ v
    R = float(np.real(np.vdot(v, Gv)))
    B = Gamma_G.imag                               # real antisymmetric handedness kernel
    T = float(np.imag(np.vdot(v, B @ v)))          # = Im(v^dagger B v), the orientation-odd signal
    a = T / R if abs(R) > 1e-12 else 0.0
    return R, T, float(np.clip(a, -1.0, 1.0))


def eigenvalue_blindness(k, n, k0, D):
    """Numerical proof that the collective decay RATES are orientation-blind. The phased-dipole
    decay matrix is Gamma(d) = D(d)^dagger Gamma_G D(d) with D(d) = diag(omega^{k_j d}) UNITARY,
    a SIMILARITY transform of the fixed array Gamma_G. Hence eig(Gamma(d)) = eig(Gamma_G) is
    d-INVARIANT: the orientation does NOT ring in any collective decay rate / bright-mode
    eigenvalue (the array-level restatement of the 50.14 B3 result). Returns the max spectral
    difference between d and N-d over a few random d (machine zero ~1e-13)."""
    N = 1 << n
    Gg = array_decay_matrix(k, n, k0, D)
    rng = np.random.default_rng(12345 + n)
    diffs = []
    for _ in range(8):
        d = C.sample_secret(N, rng)
        ref = None
        for dd in (d, (N - d) % N):
            Dm = np.diag(coset_phases(k, dd, N))
            G = Dm.conj().T @ Gg @ Dm
            rates = np.sort(np.linalg.eigvalsh(G))
            if dd == d:
                ref = rates
            else:
                diffs.append(float(np.max(np.abs(rates - ref))))
    return float(np.max(diffs))


# ===========================================================================
# independent single-copy chiral control (B1-like) - to test if the COLLECTIVE
# coupling beats independent reads
# ===========================================================================
def chiral_statistic_fast(k, d, N, k0, gamma=1.0):
    """O(M) closed form of the cascaded-chiral (D=1) handedness statistic T(d), with the
    array geometry z_j = k_j (position = frequency). Identical to readouts()[1] but O(M):

      T(d) = (2 gamma / M) Im( conj(S_s) S_c ),
      S_s = sum_j sin(k0 k_j) exp(-2 pi i k_j d / N),  S_c = sum_j cos(k0 k_j) exp(-2 pi i k_j d / N).

    Derived from Im(Gamma_G)_{jl} = gamma sin(k0 (k_j - k_l)) and v_j = exp(-2 pi i k_j d/N)/sqrt M.
    Lets the array size M be swept to 2^16 without forming the M x M matrix."""
    M = len(k)
    ang = 2.0 * np.pi * (k.astype(np.int64) * int(d) % N) / N
    p = np.exp(-1j * ang)
    gk = k0 * k.astype(float)
    Ss = np.sum(np.sin(gk) * p)
    Sc = np.sum(np.cos(gk) * p)
    return float(2.0 * gamma / M * np.imag(np.conj(Ss) * Sc))


def chiral_asymmetry_fast(k, d, N, k0, gamma=1.0):
    """O(M) (T, R_tot, a) for the rank-1 chiral channel Gamma_G = gamma w w^dagger, w_j =
    exp(i k0 k_j). R_tot(d) = v^dagger Gamma_G v = (gamma/M) |sum_j exp(-i k_j (k0 + 2 pi d/N))|^2
    (the total collective emission rate); T(d) the handedness statistic; a = T/R_tot the
    bounded chiral asymmetry. No M x M matrix -- lets the one-shot model run to large M."""
    M = len(k)
    kf = k.astype(float)
    ang_d = 2.0 * np.pi * (k.astype(np.int64) * int(d) % N) / N
    p = np.exp(-1j * ang_d)
    gk = k0 * kf
    Ss = np.sum(np.sin(gk) * p)
    Sc = np.sum(np.cos(gk) * p)
    T = float(2.0 * gamma / M * np.imag(np.conj(Ss) * Sc))
    wv = np.sum(np.exp(-1j * kf * k0) * p)         # sqrt(M) * (w^dagger v)
    R_tot = float(gamma / M * (np.abs(wv) ** 2))
    a = T / R_tot if abs(R_tot) > 1e-12 else 0.0
    return T, R_tot, float(np.clip(a, -1.0, 1.0))


def independent_statistic_fast(k, d, N):
    """O(M) independent single-copy conjugate read (B1), vectorized: sum_j (-sin(2 pi k_j d/N))
    cot(pi k_j / N). The fixed, d-independent per-copy chiral matched filter -- the control the
    collective array must beat to justify superradiance."""
    kk = k.astype(np.int64) % N
    y = -np.sin(2.0 * np.pi * (kk * int(d) % N) / N)
    h = np.where(kk == 0, 0.0, 1.0 / np.tan(np.pi * np.where(kk == 0, 1, kk) / N))
    return float(np.dot(y, h))


def independent_statistic(k, d, N):
    """The B1-style fixed single-copy conjugate read on the SAME M emitters: sum_j Im(p_j)
    h(k_j) with the fixed Hilbert kernel h(k)=cot(pi k/N). Uses NO inter-emitter coupling
    (no collective/chiral array) -- the control for whether the chiral COLLECTIVE coupling
    adds anything beyond summing independent quadratures."""
    p = coset_phases(k, d, N)
    y = p.imag                                 # = -sin(2 pi k d / N), the conjugate quadrature
    h = np.zeros(N)
    kk = np.arange(1, N)
    h[1:] = 1.0 / np.tan(np.pi * kk / N)
    return float(np.dot(y, h[k % N]))
