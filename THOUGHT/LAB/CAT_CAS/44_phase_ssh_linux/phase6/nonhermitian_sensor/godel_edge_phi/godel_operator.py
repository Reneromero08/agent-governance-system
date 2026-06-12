"""
godel_operator.py - the 6th non-Hermitian sensor for the Exp 50.14 dihedral fold:
a Godel-feedback-edge Hatano-Nelson chain whose tunable boundary-twist PHASE phi is
swept, with the point-gap winding tracked via Exp 36's rank-1 determinant lemma.

THE OPERATOR (faithful to the prompt + to Exp 36 + to the prior PT sensor)
-------------------------------------------------------------------------
Site space x in Z_N, N = 2^n (the dimension IS part of the price, like every prior
sensor: H_dim = 2^n). The public map f(x) = x if accept(x) else (x+1) mod N has a
+1 DIRECTION (broken fold) -> encode it as a non-reciprocal Hatano-Nelson chain, and
mark its fixed points {d, N-d} (= the accepted sites) as on-site SELF-LOOPS:

  base tridiagonal chain T (OPEN, no boundary bond):
      T[x, x-1] = t_R = exp(+a)      forward hop  x-1 -> x   (the +1 direction of f)
      T[x, x+1] = t_L = exp(-a)      back hop
      T[x, x]   = -i*ell + s*accept(x)   baseline dissipation + SELF-LOOP at fixed pts
                                          accept(x) is PUBLIC and fold-EVEN.

  Godel / flux edge (the single swept boundary entry, EXACTLY Exp 36's H[0,N-1]):
      H(phi)[0, N-1] = lambda * exp(i*phi)     a rank-1 perturbation  u v^T,
                                               u = e_0, v = e_{N-1}, twist phi in [0,2pi)

THE RANK-1 DETERMINANT LEMMA (Exp 36, honest classical linear algebra; NO CTC oracle)
------------------------------------------------------------------------------------
  det(E I - H(phi)) = det(E I - T) * (1 - lambda e^{i phi} v^T (E I - T)^{-1} u)
                    = D_open(E) * (1 - lambda e^{i phi} g),   g = [(E I - T)^{-1}]_{N-1,0}

For the OPEN tridiagonal T, the corner Green's function is exact and closed form:
      g = t_R^{N-1} / D_open(E)          (cofactor of the (0,N-1) corner)
so the D_open CANCELS in the swept term and the whole phi dependence is a single line:

      det(E I - H(phi)) = D_open(E) - lambda * t_R^{N-1} * e^{i phi}.

As phi sweeps 0 -> 2pi this is a CIRCLE of radius  R = lambda * t_R^{N-1}  centered at
the (instance-dependent, fold-EVEN) complex number D_open(E). The point-gap winding is

      W(E_ref) = +1   if   R > |D_open(E_ref)|     (the loop encloses 0)
               =  0   otherwise,

i.e. a log-space comparison  log R  vs  log|D_open|. This is precisely Exp 36's
"loop radius vs gap" transition (there r = lambda^{1/N}; here R = lambda * e^{a(N-1)}),
and it is computed for ALL phi at O(1) after ONE O(N) tridiagonal pass for D_open.

THE PHASE HANDLE, MECHANISTICALLY
---------------------------------
phi enters det ONLY through the additive term  -R e^{i phi}, whose amplitude
R = lambda * t_R^{N-1} is a PUBLIC CONSTANT (a function of N and the public hop a, with
no reference to d at all). The phi-loop is therefore a circle of FIXED radius that
merely ROTATES; all instance / orientation dependence sits in the CENTER D_open(E),
which is built from accept(x) and is fold-EVEN. Hence W(phi), the Berry/Zak winding
integral over phi, the loop crossing-phase, and every phi-resolved feature are FUNCTIONS
OF THE FOLD-EVEN D_open -> they read the even answer a = min(d,N-d) but are blind to the
orientation o = 1[d<N/2]. That is the predicted fold inheritance, made exact.

ASCII only. No RNG inside this module (gate / runner seed everything).
"""
import numpy as np

# ---- public operator constants (NONE reference d or the half-range) ----
A_HOP = 0.2            # non-reciprocity a: t_R = e^{+a}, t_L = e^{-a}; direction of f
ELL = 0.1             # baseline on-site dissipation (mirrors Exp 36 -i*ell diagonal)
S_LOOP = 1.0          # self-loop weight added at the accepted (fixed-point) sites
LOOP_RADIUS = 1.0     # R = lambda * t_R^{N-1}: PUBLIC loop radius (we fix R, which pins
                      # lambda = R * exp(-a(N-1)) -- the exp-small Godel coupling of
                      # Exp 36's lambda_c ~ gap^N scaling; R itself carries no d-info).
# public reference energies on the contour (all public constants):
E_REFS = np.array([0.0 + 0.0j, 0.5 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j], dtype=complex)


# ---------------------------------------------------------------------------
# public accept(x) profile over the whole ring, O(N log N) via the binned spectrum
# ---------------------------------------------------------------------------
def accept_profile(k, b, N):
    """accept(x) = score(x) > M/4 for every x in Z_N, using PUBLIC (k,b) only.
    score(x) = sum_i b_i cos(2 pi k_i x / N) = Re( sum_m chat[m] e^{2 pi i m x / N} )
    where chat[m] = sum_{i: k_i = m} b_i. Evaluated for all x at once by an inverse FFT
    (score = Re(N * ifft(chat))). O(N log N + M); never touches d."""
    M = len(b)
    chat = np.zeros(N, dtype=complex)
    np.add.at(chat, k % N, b.astype(complex))
    score_all = np.real(N * np.fft.ifft(chat))     # score(x) for x = 0..N-1
    return (score_all > (M / 4.0)).astype(float)    # 1.0 at the two fixed points {d,N-d}


# ---------------------------------------------------------------------------
# log-space tridiagonal determinant D_open(E) for a vector of reference energies
# ---------------------------------------------------------------------------
def tridet_log_vec(diag, E_grid, q, rescale_every=24):
    """For each E in E_grid, return (log|det(E I - T)|, unit phase) where T is the open
    tridiagonal with on-site diagonal `diag` (complex, length N) and constant off-diagonal
    product q = t_R t_L. Three-term recurrence D_j = (E - diag_j) D_{j-1} - q D_{j-2},
    carried in log space (periodic magnitude rescale) so the exponentially large/small
    determinant never overflows at n = 14. Vectorized over E_grid; O(N) per energy."""
    E = np.asarray(E_grid, dtype=complex)
    Dpp = np.ones_like(E)                      # D_{-1} = 1
    Dp = (E - diag[0])                         # D_0 = (E - diag_0)
    logs = np.zeros(E.shape, dtype=float)
    N = len(diag)
    for j in range(1, N):
        Dn = (E - diag[j]) * Dp - q * Dpp
        Dpp = Dp
        Dp = Dn
        if (j % rescale_every) == 0:
            m = np.abs(Dp)
            sc = np.where(m > 0.0, m, 1.0)
            Dp = Dp / sc
            Dpp = Dpp / sc
            logs = logs + np.log(sc)
    m = np.abs(Dp)
    sc = np.where(m > 0.0, m, 1.0)
    logmag = logs + np.log(sc)
    phase = np.where(m > 0.0, Dp / sc, np.ones_like(Dp))
    return logmag, phase


# ---------------------------------------------------------------------------
# phi-swept winding + phi-resolved features (closed form via the rank-1 lemma)
# ---------------------------------------------------------------------------
def phi_features(diag, E_grid, a=A_HOP, R=LOOP_RADIUS, n_phi=256, want_numeric=False):
    """Given the operator diagonal (public) and public reference energies, return the
    phi-resolved readout of the Godel-edge winding. EVERYTHING here is a function of
    D_open(E) (fold-even) and the public constants (a, R).

    For each E_ref:
      log|D_open|, arg(D_open)            -- the loop CENTER (fold-even, instance-varying)
      W = 1[ log R > log|D_open| ]         -- point-gap winding over the phi loop (0/1)
      margin = log R - log|D_open|         -- signed log-distance to the W transition
      phi_cross = arg(D_open)              -- phi at which the loop is closest to E_ref
                                              (-> cos, sin features)
    Plus a single GLOBAL feature log R = log(R) (a PUBLIC CONSTANT, identical for every
    instance -- it is literally the amplitude of the phi handle, exhibiting that the
    phase knob carries no d information).
    If want_numeric: also return the numerically integrated winding over an n_phi grid
    (the Berry/Zak winding integral) to confirm it equals the closed form."""
    q = np.exp(a) * np.exp(-a)                 # t_R t_L = 1 exactly
    logmag, phase = tridet_log_vec(diag, E_grid, q)
    logR = np.log(R) + (len(diag) - 1) * a     # log R = log lambda + (N-1) log t_R
    W = (logR > logmag).astype(float)
    margin = logR - logmag
    arg_c = np.angle(phase)
    feats = []
    for j in range(len(E_grid)):
        feats += [float(logmag[j]), float(arg_c[j]), float(W[j]), float(margin[j]),
                  float(np.cos(arg_c[j])), float(np.sin(arg_c[j]))]
    feats.append(float(logR))                  # the public-constant phi amplitude
    out = np.array(feats, dtype=float)
    if not want_numeric:
        return out
    # numeric Berry/winding integral over phi (cheap O(n_phi) per E_ref via the lemma):
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=True)
    Wnum = np.zeros(len(E_grid))
    for j in range(len(E_grid)):
        L = max(logR, logmag[j])
        center = np.exp(logmag[j] - L) * phase[j]
        rad = np.exp(logR - L)
        curve = center - rad * np.exp(1j * phis)
        ang = np.unwrap(np.angle(curve))
        Wnum[j] = (ang[-1] - ang[0]) / (2.0 * np.pi)
    return out, np.round(Wnum).astype(int), W.astype(int)


# ---------------------------------------------------------------------------
# Gate candidates: O(inst) -> 1D float feature vector  (no_smuggle_gate contract)
# ---------------------------------------------------------------------------
def O_public_godel_phi(inst):
    """THE 6th SENSOR (public). Build the Godel-edge Hatano-Nelson diagonal from the
    PUBLIC accept(x) only, then read the phi-swept point-gap winding + phi-resolved loop
    features. A pure function of (k, b, N): byte-identical under the fold -> delta == 0.
    Prediction: FAIL_CHANCE (every feature is a function of the fold-even D_open)."""
    k, b, N = inst["k"], inst["b"], inst["N"]
    acc = accept_profile(k, b, N)
    diag = (-1j * ELL) + S_LOOP * acc          # PUBLIC, fold-even self-loop diagonal
    return phi_features(diag, E_REFS)


def O_smuggle_godel_phi(inst):
    """CONSTRUCTION-NATIVE SMUGGLE (positive control). Identical sensor, EXCEPT the
    self-loop / edge set is placed using the HIDDEN d: a self-loop is put at the TRUE
    secret site d ONLY (not at its fold partner N-d). This makes the operator's edge set
    a function of which half d lives on, so D_open becomes fold-ODD: the readout flips
    under d <-> N-d. The hardened gate MUST flag this FAIL_SMUGGLE (exact d-invariance
    delta > 0, and orientation AUC -> 1). This is exactly the prompt's trap: 'if reading
    o requires the edge set from d, that is FAIL_SMUGGLE'."""
    k, b, N = inst["k"], inst["b"], inst["N"]
    d = int(inst["d"]) % N                      # <-- reads the secret
    acc = accept_profile(k, b, N)               # public part (fold-even)
    diag = (-1j * ELL) + S_LOOP * acc
    diag[d] = (-1j * ELL) + 2.0 * S_LOOP        # <-- asymmetric self-loop at d ONLY
    return phi_features(diag, E_REFS)


def O_useless_even(inst):
    """KNOWN-USELESS even control (must be FAIL_CHANCE): pure public moments of the
    accept profile and D_open at one energy, with the winding stripped. Fold-even."""
    k, b, N = inst["k"], inst["b"], inst["N"]
    acc = accept_profile(k, b, N)
    diag = (-1j * ELL) + S_LOOP * acc
    logmag, phase = tridet_log_vec(diag, E_REFS[:1], 1.0)
    return np.array([float(np.mean(acc)), float(np.sum(acc)), float(logmag[0])])
