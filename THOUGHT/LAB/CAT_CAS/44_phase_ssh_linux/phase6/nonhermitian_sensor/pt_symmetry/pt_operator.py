"""
pt_operator.py - PT-symmetric / biorthogonal non-Hermitian sensor for the Exp 50.14 fold.

APPROACH (Phase 6 / nonhermitian_sensor / pt_symmetry):
Map the construction fold sigma: d -> N-d to the PARITY operator P: x -> (N-x) mod N
on the site ring Z_N, pair it with complex conjugation T, and ask whether any spectral /
topological / biorthogonal readout of a non-Hermitian operator built from PUBLIC data
(k, b) can read the orientation bit o = 1[d < N/2].

Site-space operators (dimension = N = 2^n; the dimension itself is part of the price):

  H_public = t_R * S + t_L * S^T + i * gamma * diag(g_even)
      g_even(x) = score(x)/M = (1/M) sum_i b_i cos(2 pi k_i x / N)     [PUBLIC, even]
      t_R = exp(+a), t_L = exp(-a): non-reciprocal hopping encoding the +1 direction
      of the public map f (Hatano-Nelson directionality). S[x, x-1] = 1 (ring shift).
      Note g_even is parity-EVEN (score(N-x) = score(x)), so the anti-Hermitian part
      is parity-symmetric: gain at BOTH d and N-d. A genuinely PT-symmetric gain/loss
      profile with P = the fold would have to be parity-ODD - which is exactly the
      absent quadrature channel. That is the sharp PT statement of the wall.

  H_smuggle = S + S^T + i * gamma_s * diag(g_odd)        [CONTROL - reads hidden d]
      g_odd(x) = (1/M) sum_i sin(2 pi k_i d / N) sin(2 pi k_i x / N)
      gain at x = d, loss at x = N-d: parity-ODD profile, the genuinely PT-symmetric
      operator. It requires the hidden quadrature channel sin(2 pi k d / N), which IS
      the smuggle; it exists here only as a positive control for the instrument.

Structure facts the runner confirms numerically:
  (T1) H_public is a pure function of (k, b, N): under the fold at fixed public data
       it is BYTE-IDENTICAL, so every readout has fold-delta == 0 and AUC == 1/2.
  (T2) P H(g) P^-1 = H(-g) for symmetric hopping (exact entrywise similarity), so even
       for the SMUGGLE operator the SPECTRUM (PT-broken fraction, EP structure, point-
       gap winding, det) is fold-blind. Orientation lives ONLY in the eigenvectors
       (WHICH site hosts the gain mode) - a biorthogonal observable, not a PT phase.
  (T3) The point-gap winding of H_public is a genuine nonzero non-Hermitian invariant
       (the directionality of f DOES survive into the topology), but it is the SAME
       constant for every instance: it reads the public +1 direction of the walk, not
       the hidden half.

Point-gap winding, computed honestly: thread flux phi through ONE bond, then
  det(E0 - H(phi)) = K0(E0) - t_R^N e^{i phi} - t_L^N e^{-i phi}
  K0(E0) = D_open - t_R t_L D_inner    (two O(N) tridiagonal recursions, log-scaled)
  W(E0) = winding number of det(E0 - H(phi)) around 0 as phi sweeps 0 -> 2 pi.
Cost is O(N) = O(2^n) per evaluation: the site space is exponential-dimensional. That
cost is part of the measurement, not an implementation accident.

ASCII only. No RNG inside this module.
"""
import numpy as np

A_HOP = 0.2          # non-reciprocity: t_R = e^{+a}, t_L = e^{-a} (direction of f)
GAMMA_PUB = 1.0      # public gain strength
GAMMA_SMUG = 2.0     # control gain strength
E0_IN = 0.0 + 0.0j   # base energy inside the Hatano-Nelson loop (public constant)
E0_OUT = 3.5 + 0.0j  # base energy outside the spectrum (public constant)


# ---------------------------------------------------------------------------
# gain/loss profiles
# ---------------------------------------------------------------------------
def g_even_profile(k, b, N):
    """PUBLIC even profile: g(x) = score(x)/M for all x in Z_N. O(N*M)."""
    M = len(b)
    x = np.arange(N)
    return (np.cos(2.0 * np.pi * np.outer(x, k) / N) @ b) / M


def g_odd_profile(k, d, N):
    """HIDDEN odd profile (CONTROL ONLY - reads d): gain at d, loss at N-d.
    Uses the quadrature channel sin(2 pi k d / N) that public data provably lacks."""
    M = len(k)
    s_hidden = np.sin(2.0 * np.pi * k * d / N)
    x = np.arange(N)
    return (np.sin(2.0 * np.pi * np.outer(x, k) / N) @ s_hidden) / M


# ---------------------------------------------------------------------------
# dense operator (for n <= 10 spectral readouts)
# ---------------------------------------------------------------------------
def build_dense(N, g, a, gamma):
    """H = e^{+a} S + e^{-a} S^T + i gamma diag(g), with S[x, x-1] = 1 on the ring."""
    tR = float(np.exp(a))
    tL = float(np.exp(-a))
    H = np.zeros((N, N), dtype=complex)
    idx = np.arange(N)
    H[idx, (idx - 1) % N] = tR        # hop x-1 -> x : the +1 direction of f
    H[idx, (idx + 1) % N] = tL
    H[idx, idx] = 1j * gamma * np.asarray(g)
    return H


# ---------------------------------------------------------------------------
# O(N) point-gap winding via the corner expansion of det(E - H(phi))
# ---------------------------------------------------------------------------
def _tridet_log(vals, q):
    """det of tridiagonal matrix with diagonal vals and off-diagonal pair product q,
    via D_j = v_j D_{j-1} - q D_{j-2}, log-rescaled. Returns (logmag, unit_phase)."""
    Dp = 1.0 + 0.0j
    Dpp = 0.0 + 0.0j
    logs = 0.0
    for v in vals:
        Dn = v * Dp - q * Dpp
        Dpp, Dp = Dp, Dn
        m = abs(Dp)
        if m > 1e120 or (0.0 < m < 1e-120):
            Dp /= m
            Dpp /= m
            logs += np.log(m)
    m = abs(Dp)
    if m == 0.0:
        return -np.inf, 1.0 + 0.0j
    return logs + np.log(m), Dp / m


def _logdet_K0(E, diag_vals, q):
    """K0(E) = D_open - q * D_inner for the tridiagonal part of (E - H), log form.
    diag_vals = the diagonal of H (complex array). q = t_R * t_L > 0."""
    dv = (E - np.asarray(diag_vals)).tolist()
    N = len(dv)
    lo, uo = _tridet_log(dv, q)
    li, ui = _tridet_log(dv[1:N - 1], q)
    lqi = li + np.log(q)
    L = max(lo, lqi)
    if L == -np.inf:
        return -np.inf, 1.0 + 0.0j
    val = np.exp(lo - L) * uo - np.exp(lqi - L) * ui
    m = abs(val)
    if m == 0.0:
        return -np.inf, 1.0 + 0.0j
    return L + np.log(m), val / m


def winding(E0, diag_vals, a, n_phi=512):
    """Point-gap winding W(E0): winding number of det(E0 - H(phi)) around 0 as flux
    phi sweeps one bond, 0 -> 2 pi. Exact corner expansion, evaluated in log space:
        det = K0 - t_R^N e^{i phi} - t_L^N e^{-i phi}
    Returns (W, log|K0|). Cost O(N + n_phi); N = 2^n is the honest exponential price."""
    N = len(diag_vals)
    tR = float(np.exp(a))
    tL = float(np.exp(-a))
    lk0, u0 = _logdet_K0(E0, diag_vals, tR * tL)
    lp = N * np.log(tR)
    lm = N * np.log(tL)
    L = max(lk0, lp, lm)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=True)
    curve = (np.exp(lk0 - L) * u0
             - np.exp(lp - L) * np.exp(1j * phi)
             - np.exp(lm - L) * np.exp(-1j * phi))
    ang = np.unwrap(np.angle(curve))
    W = int(np.round((ang[-1] - ang[0]) / (2.0 * np.pi)))
    return W, float(lk0)


def validate_dets(seed=12345):
    """Validate the corner expansion against dense numpy determinants at small N.
    Returns the worst relative error over a grid of (N, a, E, phi)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for N in (6, 9, 16):
        for a in (0.0, 0.2):
            g = rng.normal(size=N)
            H0 = build_dense(N, g, a, 1.0)
            tR = float(np.exp(a))
            tL = float(np.exp(-a))
            dg = np.diag(H0).copy()
            for E in (0.3 + 0.1j, -0.7 + 0.4j, 2.5 + 0.0j):
                lk0, u0 = _logdet_K0(E, dg, tR * tL)
                K0 = np.exp(lk0) * u0
                for phi in (0.0, 1.1, 2.7):
                    Hp = H0.copy()
                    Hp[0, N - 1] = tR * np.exp(1j * phi)
                    Hp[N - 1, 0] = tL * np.exp(-1j * phi)
                    dd = np.linalg.det(E * np.eye(N) - Hp)
                    mine = (K0 - (tR ** N) * np.exp(1j * phi)
                            - (tL ** N) * np.exp(-1j * phi))
                    rel = abs(dd - mine) / max(abs(dd), 1e-12)
                    worst = max(worst, rel)
    return float(worst)


# ---------------------------------------------------------------------------
# feature extractors (continuous everywhere: no knife-edge counts)
# ---------------------------------------------------------------------------
def spectral_feats(w):
    """Order parameters of the PT story from eigenvalues only: max/mean/std of Im
    (PT-breaking), soft complex fraction, top-2 splitting (EP proximity of the gain
    doublet), min nearest spacing (global EP proximity proxy)."""
    im = w.imag
    re = w.real
    order = np.argsort(-im)
    top2 = float(abs(w[order[0]] - w[order[1]])) if len(w) > 1 else 0.0
    sr = np.sort_complex(w)
    minsp = float(np.min(np.abs(np.diff(sr)))) if len(w) > 1 else 0.0
    softfrac = float(np.mean(np.tanh(np.abs(im) / 0.05)))
    return [float(im.max()), float(im.min()), float(im.mean()), float(im.std()),
            softfrac, top2, float(re.max()), float(re.min()), minsp]


def evec_feats(w, V, N):
    """Biorthogonal readout of the dominant-gain mode: circular centroid (the Im
    component is the would-be orientation reader: sin(2 pi d / N) > 0 iff d < N/2),
    right IPR, biorthogonal IPR."""
    j = int(np.argmax(w.imag))
    r = V[:, j]
    p = np.abs(r) ** 2
    p = p / p.sum()
    z = complex(np.sum(p * np.exp(2j * np.pi * np.arange(N) / N)))
    ipr_r = float(np.sum(p ** 2))
    Vinv = np.linalg.inv(V)
    lv = Vinv[j, :]
    br = np.abs(lv * r)
    br = br / max(br.sum(), 1e-300)
    ipr_b = float(np.sum(br ** 2))
    return [float(z.real), float(z.imag), float(abs(z)), ipr_r, ipr_b]


# ---------------------------------------------------------------------------
# Gate candidates: O(inst) -> feature vector  (contract of no_smuggle_gate)
# ---------------------------------------------------------------------------
def O_public_full(inst):
    """PUBLIC PT/biorthogonal readout, full feature set (dense eig + winding).
    Uses only (k, b, N). H dimension = 2^n."""
    k, b, N = inst["k"], inst["b"], inst["N"]
    g = g_even_profile(k, b, N)
    H = build_dense(N, g, A_HOP, GAMMA_PUB)
    w, V = np.linalg.eig(H)
    f = spectral_feats(w) + evec_feats(w, V, N)
    dg = np.diag(H).copy()
    W1, lk1 = winding(E0_IN, dg, A_HOP)
    W2, lk2 = winding(E0_OUT, dg, A_HOP)
    return np.array(f + [float(W1), float(W2), lk1, lk2])


def O_public_evals(inst):
    """PUBLIC readout, eigenvalues + winding (no eigenvectors; n = 10 budget)."""
    k, b, N = inst["k"], inst["b"], inst["N"]
    g = g_even_profile(k, b, N)
    H = build_dense(N, g, A_HOP, GAMMA_PUB)
    w = np.linalg.eigvals(H)
    f = spectral_feats(w)
    dg = 1j * GAMMA_PUB * g
    W1, lk1 = winding(E0_IN, dg, A_HOP)
    W2, lk2 = winding(E0_OUT, dg, A_HOP)
    return np.array(f + [float(W1), float(W2), lk1, lk2])


def O_public_winding(inst):
    """PUBLIC topological readout only (n = 12, 14 budget): point-gap windings and
    log|K0| at two public base energies. O(N) exact - no dense matrix is built."""
    k, b, N = inst["k"], inst["b"], inst["N"]
    g = g_even_profile(k, b, N)
    dg = 1j * GAMMA_PUB * g
    W1, lk1 = winding(E0_IN, dg, A_HOP)
    W2, lk2 = winding(E0_OUT, dg, A_HOP)
    return np.array([float(W1), float(W2), lk1, lk2])


def O_smuggle_evals(inst):
    """CONTROL: genuinely PT-symmetric operator from the HIDDEN odd channel,
    eigenvalue-only readout. Fact T2 predicts AUC ~ 0.5 and fold-delta ~ 0 (the
    spectrum is similarity-blind): even WITH the quadrature channel, the PT order
    parameter (real vs complex spectrum, EP location) cannot read orientation."""
    k, N, d = inst["k"], inst["N"], inst["d"]
    g = g_odd_profile(k, d, N)
    H = build_dense(N, g, 0.0, GAMMA_SMUG)
    w = np.linalg.eigvals(H)
    return np.array(spectral_feats(w))


def O_smuggle_evecs(inst):
    """CONTROL: same PT-symmetric operator, BIORTHOGONAL eigenvector readout (gain-
    mode centroid). Reads o (AUC -> 1) and MUST be flagged FAIL_SMUGGLE by the gate:
    the gain profile is built from sin(2 pi k d / N), i.e. from the hidden d."""
    k, N, d = inst["k"], inst["N"], inst["d"]
    g = g_odd_profile(k, d, N)
    H = build_dense(N, g, 0.0, GAMMA_SMUG)
    w, V = np.linalg.eig(H)
    return np.array(spectral_feats(w) + evec_feats(w, V, N))
