"""
candidates.py - Stage 3 quadrature-synthesis attack battery (Phase 6).

Each candidate is a map O(instance) -> 1D float feature vector, intended to
expose the orientation bit b_orient = 1[d < N/2] from PUBLIC data only.

Contract (from no_smuggle_gate.py):
    instance = {"k": int[], "b": float[], "N": int, "d": int (HIDDEN), "n": int}
A NON-smuggling candidate must read ONLY k, b, N, n.  Any candidate that touches
inst["d"] (directly or via the hidden sin channel) is smuggling and the gate's
exact d-invariance audit will catch it (max_fold_delta > 0).

The public data has E[b_i] = cos(2*pi*k_i*d/N): an EVEN function of d. The
orientation bit lives in the ODD channel sin(2*pi*k*d/N), which integrates to
zero over the public even data. Every honest candidate here is a transform of the
even (cosine) channel; the dihedral barrier predicts they all stay at chance.
Candidate #5 (Gerchberg-Saxton with a support constraint) is the designated
SMUGGLE candidate: the support constraint d in [1,N/2) IS the orientation, so we
test whether the gate catches the smuggle and report exactly where it enters.

ASCII only. All RNGs seeded by the caller (gate passes none; candidates that need
randomness derive a DETERMINISTIC seed from the public data so the d-invariance
audit -- which calls O twice on identical public data -- is exact).

Author: Stage 3 agent. Claim ceiling L4-5.
"""
import numpy as np

import construction as C


# ---------------------------------------------------------------------------
# Shared public-data primitives (NONE of these read inst["d"])
# ---------------------------------------------------------------------------
def _cos_hat(k, b, N):
    """Bin the random (k,b) onto the Z_N frequency grid and estimate
    cos_hat[m] = E[b | k == m] for m = 0..N-1.  This is the empirical even
    (cosine) spectrum.  Unvisited bins -> 0 (the unbiased prior mean).  Pure
    function of public (k,b,N).  cos_hat[m] estimates cos(2*pi*m*d/N).
    """
    cos_hat = np.zeros(N, dtype=float)
    counts = np.zeros(N, dtype=float)
    np.add.at(cos_hat, k, b)
    np.add.at(counts, k, 1.0)
    nz = counts > 0
    cos_hat[nz] /= counts[nz]
    return cos_hat, counts


def _public_seed(k, b, N):
    """Deterministic seed derived ONLY from public data, so any randomized
    candidate yields byte-identical output on inst and folded_instance(inst)
    (which share the same k,b,N).  Touching d here would smuggle, so we do not."""
    h = (int(np.sum(k.astype(np.int64)) * 1000003)
         ^ int(np.sum((b > 0).astype(np.int64)) * 2654435761)
         ^ (int(N) * 40503))
    return h & 0x7FFFFFFF


def _score_profile(k, b, N, n_probes):
    """Public matched-filter score(x) sampled on a grid of x in [0, N).
    score(x) = sum_i b_i cos(2 pi k_i x / N); pure public (k,b)."""
    xs = np.arange(N) if n_probes >= N else np.linspace(0, N, n_probes, endpoint=False)
    prof = np.array([C.score(k, b, float(x), N) for x in xs])
    return xs, prof


# ===========================================================================
# CANDIDATE 1 - Discrete Hilbert / analytic-signal on the binned cosine spectrum
# ===========================================================================
def O_hilbert_analytic(inst):
    """Bin (k,b) -> cos_hat[m] ~ cos(2*pi*m*d/N) on the Z_N grid, then apply the
    finite (DFT) Hilbert transform: multiply the conjugate-domain coefficients by
    -i*sign(freq) to synthesize a quadrature (sin-like) estimate sin_hat[m].

    MECHANISM UNDER TEST: cos_hat is an EVEN sequence in m (cos((N-m)*..)=cos(m*..)),
    because both m and N-m are visited symmetrically by random k.  The Hilbert
    transform of a real even sequence is real ODD -- but the binning estimates the
    SAME even cos for m and N-m, so the synthesized 'sin' is forced odd-symmetric
    AROUND the grid with NO absolute sign reference to d.  We read it at fixed low
    rungs.  Prediction: chance (the sign of d's half is not recoverable; only |sin|
    structure survives the even binning)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    cos_hat, counts = _cos_hat(k, b, N)
    # finite Hilbert transform via DFT: H[x] = IDFT( -i*sign(f) * DFT(x) )
    F = np.fft.fft(cos_hat)
    f = np.fft.fftfreq(N) * N           # integer-ish frequency bins
    mult = -1j * np.sign(f)
    sin_hat = np.real(np.fft.ifft(F * mult))   # the analytic-signal quadrature part
    # read the synthesized quadrature at the dyadic low rungs (where odd channel lives)
    rungs = [1, 2, 3, 4, 8]
    feats = [float(sin_hat[r % N]) for r in rungs]
    # also the analytic-signal phase at rung 1 (atan2(sin_hat, cos_hat))
    feats.append(float(np.arctan2(sin_hat[1 % N], cos_hat[1 % N])))
    return np.array(feats)


# ===========================================================================
# CANDIDATE 2 - Double-angle / dyadic ladder coupling
# ===========================================================================
def O_double_angle(inst):
    """Use cos(2t)=2cos^2 t - 1 and sin(2t)=2 sin t cos t to try to PIN sin at a
    rung from cosines at related frequencies.

    cos_hat[k] ~ cos(theta_k) with theta_k = 2*pi*k*d/N.  From cos(theta_k) alone,
    |sin(theta_k)| = sqrt(1 - cos^2) is fixed but the SIGN is the missing bit.  The
    double-angle identity gives cos(theta_{2k}) = 2 cos^2(theta_k) - 1, which is a
    CONSISTENCY relation, not a sign source -- it is even in d at every rung.

    The only way the algebra could leak a sign is sin(theta_{2k}) = 2 s c with
    s=sin(theta_k), c=cos(theta_k): but sin(theta_{2k}) is ALSO an odd channel we do
    not have.  So we can build |sin| at each rung and the PRODUCT/CONSISTENCY
    residuals, all even.  Prediction: chance.  We expose |sin| at rungs, the
    double-angle residual, and a ladder-product feature."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    cos_hat, _ = _cos_hat(k, b, N)
    ladder = [1 << j for j in range(0, n)]      # 1,2,4,...,N/2
    c = np.array([cos_hat[(r) % N] for r in ladder])
    c = np.clip(c, -1.0, 1.0)
    abs_sin = np.sqrt(np.maximum(0.0, 1.0 - c * c))   # |sin(theta_r)|, EVEN in d
    feats = list(abs_sin)
    # double-angle consistency residual: cos(theta_{2r}) - (2 c_r^2 - 1)
    for j in range(len(ladder) - 1):
        pred = 2.0 * c[j] * c[j] - 1.0
        feats.append(float(c[j + 1] - pred))
    # half-angle sign attempt (still even): cos(theta_1) routed through sqrt((1+c)/2)
    feats.append(float(np.sqrt(max(0.0, (1.0 + c[0]) / 2.0))))
    return np.array(feats)


# ===========================================================================
# CANDIDATE 3 - Bispectrum (third-order) feature
# ===========================================================================
def O_bispectrum(inst):
    """Third-order correlations B(k1,k2)=X[k1]X[k2]conj(X[k1+k2]) can in principle
    expose phase coupling that the power spectrum (2nd order) cannot.

    We form the empirical complex spectrum from PUBLIC data only.  Crucially we have
    NO sin channel, so the best public complex estimate is X[m] = cos_hat[m] + 0j
    (the imaginary part is exactly the absent channel; setting it to 0 is the honest
    public estimate).  The bispectrum of a REAL-EVEN spectrum is real -- its phase is
    0 or pi, carrying magnitude coupling but no absolute orientation sign.  We expose
    real and imaginary bispectral features on dyadic triples (k, k, 2k) and
    (k, N/2-k, N/2).  Prediction: chance (bispectral PHASE of an even sequence cannot
    encode the odd orientation bit)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    cos_hat, _ = _cos_hat(k, b, N)
    X = cos_hat.astype(complex)          # imaginary part = absent odd channel -> 0
    feats = []
    rungs = [1, 2, 3, 5]
    for r in rungs:
        k1, k2 = r, r
        B = X[k1 % N] * X[k2 % N] * np.conj(X[(k1 + k2) % N])
        feats.append(float(np.real(B)))
        feats.append(float(np.imag(B)))         # 0 for even X -> the tell
    # a mixed triple touching the Nyquist fold (N/2): (r, N/2 - r, N/2)
    for r in [1, 2]:
        k1, k2 = r, (N // 2 - r) % N
        B = X[k1 % N] * X[k2 % N] * np.conj(X[(k1 + k2) % N])
        feats.append(float(np.real(B)))
        feats.append(float(np.imag(B)))
    return np.array(feats)


# ===========================================================================
# CANDIDATE 4 - Autocorrelation / Wiener-Khinchin + x->-x asymmetry
# ===========================================================================
def O_autocorr_asym(inst):
    """Reconstruct the public score profile score(x) over x in Z_N, then probe its
    asymmetry under x -> -x (== N - x), which is the d <-> N-d fold in the x domain.

    score(x) = sum_i b_i cos(2 pi k_i x / N) is an EVEN function of x (cos is even),
    peaking at BOTH x=d and x=N-d.  So score(x) - score(N-x) == 0 identically (up to
    sampling), i.e. the x-domain profile is symmetric and the fold asymmetry is
    structurally zero.  We expose: the antisymmetric part of the profile at several
    x, the autocorrelation (Wiener-Khinchin via |FFT(profile)|^2) and its odd part.
    Prediction: chance (the antisymmetric part is ~0 by construction; the
    orientation cannot survive an even profile)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    # Evaluate on the FULL integer grid x = 0..N-1 so the fold x -> (N - x) mod N is
    # the exact circular reversal np.roll(prof[::-1], 1); this makes the
    # antisymmetric-part claim honest (no off-by-one from a coarse linspace grid).
    xs = np.arange(N)
    cos_table = np.cos(2 * np.pi * np.outer(k, xs) / N)   # M x N
    prof = b @ cos_table                                  # score(x) for all x, O(MN)
    # antisymmetric part under x -> (N - x) mod N (the fold in the x-domain)
    prof_rev = np.roll(prof[::-1], 1)     # index x -> (N - x) mod N
    anti = (prof - prof_rev) / 2.0
    feats = []
    # antisymmetric profile sampled at a few indices (should be ~0)
    idx = np.linspace(1, len(prof) - 1, 8).astype(int)
    feats.extend([float(anti[i]) for i in idx])
    feats.append(float(np.mean(np.abs(anti))))
    feats.append(float(np.max(np.abs(anti))))
    # Wiener-Khinchin autocorrelation and its odd lag part
    P = np.abs(np.fft.fft(prof)) ** 2
    ac = np.real(np.fft.ifft(P))
    odd_ac = (ac - np.roll(ac[::-1], 1)) / 2.0   # circular lag reversal
    feats.append(float(np.mean(np.abs(odd_ac))))
    return np.array(feats)


# ===========================================================================
# CANDIDATE 5 - Gerchberg-Saxton phase retrieval WITH support constraint
#               (the DESIGNATED SMUGGLE candidate)
# ===========================================================================
def O_gerchberg_saxton_PUBLIC(inst):
    """Phase retrieval via alternating projection, PUBLIC-ONLY version.

    We have magnitudes (cos_hat, the real spectrum) and want to recover phase.  An
    honest public support constraint cannot reference d.  Here we use a SYMMETRIC
    support (the only public-justifiable one): the signal is real and lives on Z_N
    with no half-plane preference.  Alternating between 'spectrum matches cos_hat'
    and 'signal is real' converges to a real-even fixed point -> synthesized phase is
    0/pi only.  Prediction: chance (no orientation; the symmetric support is even)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    cos_hat, _ = _cos_hat(k, b, N)
    seed = _public_seed(k, b, N)
    rng = np.random.default_rng(seed)
    mag = np.abs(cos_hat)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=N))   # random init (public seed)
    X = mag * phase
    for _ in range(60):
        x = np.fft.ifft(X)
        x = np.real(x)                       # real-signal constraint (public, even)
        X2 = np.fft.fft(x)
        ph = np.exp(1j * np.angle(X2))
        X = mag * ph                         # impose known magnitudes
    sin_est = np.imag(np.fft.fft(np.real(np.fft.ifft(X))))
    rungs = [1, 2, 3, 4]
    return np.array([float(sin_est[r % N]) for r in rungs])


def O_gerchberg_saxton_SMUGGLE(inst):
    """Phase retrieval via alternating projection, SUPPORT-CONSTRAINED version.

    DESIGNATED SMUGGLE.  The 'support constraint d in [1, N/2)' IS the orientation
    bit.  To impose it, the iteration must know which half d lives on -- and the ONLY
    place that information exists is inst['d'].  We deliberately read it to build a
    half-plane support mask, exactly as a naive phase-retrieval port would.  This
    lifts the AUC (it literally injects the answer) but the gate's exact d-invariance
    audit MUST flag it: on folded_instance the mask flips, so O's output changes ->
    max_fold_delta > 0 -> FAIL_SMUGGLE.  We report precisely where the smuggle enters:
    the support mask `lower_half` reads inst['d'].
    """
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    cos_hat, _ = _cos_hat(k, b, N)
    seed = _public_seed(k, b, N)
    rng = np.random.default_rng(seed)
    mag = np.abs(cos_hat)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=N))
    X = mag * phase

    # --- THE SMUGGLE: support mask references the hidden d's half ---------------
    d = inst["d"]                                  # <-- reading the secret
    lower_half = int(d % N) < (N / 2)              # <-- this IS the orientation bit
    support = np.ones(N, dtype=float)
    if lower_half:
        support[N // 2:] = 0.0                      # keep lower-half support
    else:
        support[:N // 2] = 0.0                      # keep upper-half support
    # ---------------------------------------------------------------------------

    for _ in range(60):
        x = np.fft.ifft(X)
        x = np.real(x) * support                   # support projection uses the smuggle
        X2 = np.fft.fft(x)
        X = mag * np.exp(1j * np.angle(X2))
    sin_est = np.imag(np.fft.fft(np.real(np.fft.ifft(X))))
    rungs = [1, 2, 3, 4]
    return np.array([float(sin_est[r % N]) for r in rungs])


# ===========================================================================
# CANDIDATE 6 - Half-angle / Chebyshev sign-lift (additional principled attack)
# ===========================================================================
def O_halfangle_chebyshev(inst):
    """Additional principled attack.  The orientation bit is sign(sin(2*pi*d/N)).
    From the public cos channel we know cos(theta_k) at every rung; the half-angle
    formula sin(theta) = sign * sqrt((1-cos(2 theta))/2) needs the SIGN, which is the
    missing bit.  We instead try to lift the sign via the Chebyshev recursion that
    links cos(k theta) across ALL rungs: T_k(cos theta) = cos(k theta).  Inverting
    this system pins |theta| but the recursion is invariant under theta -> -theta
    (== d -> N-d), so any solution is sign-ambiguous by construction.

    We expose: the recovered cos(theta_1) two ways (direct bin vs Chebyshev-inverted
    from cos(theta_2),cos(theta_4)), their residual, and a sqrt-half-angle magnitude.
    Prediction: chance (the Chebyshev/Newton lift is even in d -- no branch selects
    the orientation without external sign info)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    cos_hat, _ = _cos_hat(k, b, N)
    ladder = [1 << j for j in range(0, min(n, 6))]
    c = np.clip(np.array([cos_hat[r % N] for r in ladder]), -1.0, 1.0)
    feats = []
    # Chebyshev-consistency: cos(2 theta) should equal 2 c0^2 - 1, cos(4theta)=T4(c0)...
    if len(c) >= 2:
        feats.append(float(c[1] - (2 * c[0] ** 2 - 1)))           # T2 residual
    if len(c) >= 3:
        feats.append(float(c[2] - (8 * c[0] ** 4 - 8 * c[0] ** 2 + 1)))  # T4 residual
    # half-angle magnitude (even): |sin(theta_1/2)|
    feats.append(float(np.sqrt(max(0.0, (1.0 - c[0]) / 2.0))))
    feats.append(float(np.sqrt(max(0.0, (1.0 + c[0]) / 2.0))))
    # |sin| at each rung
    feats.extend([float(np.sqrt(max(0.0, 1.0 - cc * cc))) for cc in c])
    return np.array(feats)
