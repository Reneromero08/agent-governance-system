"""
hn_operator.py - Hatano-Nelson non-reciprocal hopping sensor for the Exp 50.14 public
construction, plus the non-Hermitian topological invariants read from it.

KEY HYPOTHESIS under test: the public map f(x)=x if accept(x) else (x+1) mod N has a
DIRECTION (the +1). Directed / non-reciprocal dynamics are what a non-Hermitian operator
encodes. We encode +1 as ASYMMETRIC hopping g on the ring Z_N and the public score
landscape as the on-site potential V_x = -gamma*score(x)/M (wells at the fixed points
{d, N-d}). We then read invariants with NO Hermitian analog and ask the no-smuggle gate
whether any reads the orientation bit o = 1[d < N/2] that the fold sigma destroys.

CENTRAL ALGEBRAIC FACT (proved in gauge_equivalence_residual): under OPEN boundaries the
HN operator is an EXACT imaginary-gauge similarity of a HERMITIAN tridiagonal,
    H_obc = S H_herm S^{-1},   S = diag(e^{g x}),   H_herm symmetric, hopping t, diag V_x.
So the directionality g is a GAUGE: it leaves every spectral invariant unchanged and only
multiplies eigenvectors by the fixed envelope e^{g x} (the non-Hermitian skin pile-up).
ALL instance/data information in H enters through V_x = -gamma*score(x), the EVEN public
landscape (score(x)==score(N-x) exactly). Hence every invariant of H is a function of the
even channel => fold-invariant => CANNOT read o. The +1 directionality is real but
information-free; this is the operator-level reason the wall holds for any public-only H.

No-smuggle discipline: every quantity is a pure function of PUBLIC (k,b,N). g, gamma, t are
FIXED global constants (the public +1 and a public coupling), never d or the range. Hence H
is byte-identical under d <-> N-d at fixed public data; the gate verifies delta == 0.

ASCII only. Deterministic (fixed ARPACK start vector). N = 2^n sites => EXPONENTIAL lattice;
cost is measured by run_hn.py (the make-or-break).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Public score landscape -> the on-site potential. Pure function of (k,b,N).
# ---------------------------------------------------------------------------
def score_landscape(k, b, N, chunk=4096):
    """s[x] = (1/M) sum_i b_i cos(2 pi k_i x / N), x=0..N-1. Cost O(N*M)=O(N^1.5).
    Chunked over x to bound memory. Even: s[x] == s[N-x] exactly (the fold)."""
    k = np.asarray(k, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = b.size
    s = np.empty(N, dtype=np.float64)
    two_pi_over_N = 2.0 * np.pi / N
    for x0 in range(0, N, chunk):
        x1 = min(x0 + chunk, N)
        xs = np.arange(x0, x1, dtype=np.float64)[:, None]
        s[x0:x1] = np.cos(two_pi_over_N * xs * k[None, :]) @ b
    return s / max(M, 1)


# ---------------------------------------------------------------------------
# The Hatano-Nelson operator on Z_N (the ring = the natural domain of f).
# ---------------------------------------------------------------------------
def build_HN(s, g=0.6, t=1.0, gamma=6.0, bc="pbc", phi=0.0):
    """Sparse N x N HN matrix.
      diag      H[x,x]   = -gamma*s[x]                     (public wells)
      right hop H[x+1,x] = t*exp(+g)                       (the public +1 direction)
      left  hop H[x,x+1] = t*exp(-g)
    bc='pbc' closes the ring with flux phi: H[0,N-1]=t e^{+g} e^{+i phi},
    H[N-1,0]=t e^{-g} e^{-i phi}. bc='obc' drops the closing bond (-> skin effect).
    g,gamma,t are FIXED constants (no d, no range) => fold-invariant operator."""
    N = s.size
    tr = t * np.exp(+g)
    tl = t * np.exp(-g)
    diag = (-gamma * s).astype(np.complex128)
    rows = [np.arange(N)]; cols = [np.arange(N)]; vals = [diag]
    idx = np.arange(N - 1)
    rows.append(idx + 1); cols.append(idx); vals.append(np.full(N - 1, tr, dtype=np.complex128))
    rows.append(idx);     cols.append(idx + 1); vals.append(np.full(N - 1, tl, dtype=np.complex128))
    if bc == "pbc":
        rows.append(np.array([0]));     cols.append(np.array([N - 1])); vals.append(np.array([tr * np.exp(1j * phi)]))
        rows.append(np.array([N - 1])); cols.append(np.array([0]));     vals.append(np.array([tl * np.exp(-1j * phi)]))
    R = np.concatenate(rows); Cc = np.concatenate(cols); V = np.concatenate(vals)
    return sp.csc_matrix((V, (R, Cc)), shape=(N, N))


def hermitian_backbone(s, t=1.0, gamma=6.0):
    """The Hermitian tridiagonal H_herm with H_obc = S H_herm S^{-1}, S=diag(e^{g x}):
    symmetric hopping t, diagonal V_x = -gamma*s[x]. Well-conditioned (unlike H_obc,
    whose condition number ~ e^{g N} makes direct diagonalization unstable). Carries ALL
    the data in H; it is EVEN (fold-invariant)."""
    N = s.size
    diag = (-gamma * s).astype(np.float64)
    off = np.full(N - 1, t, dtype=np.float64)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def gauge_equivalence_residual(s, g=0.6, t=1.0, gamma=6.0):
    """Confirm the imaginary-gauge THEOREM by ALGEBRA (matrix multiply, exact - no
    eigendecomp): max|S^{-1} H_obc S - H_herm|, S=diag(e^{g x}). ~0 to machine precision.
    Only call at small N (e^{g N} must not overflow). Proves g is a gauge, not data."""
    N = s.size
    H = build_HN(s, g=g, t=t, gamma=gamma, bc="obc").toarray()
    sdiag = np.exp(g * np.arange(N, dtype=np.float64))
    Hg = (H * sdiag[None, :]) / sdiag[:, None]
    Hh = hermitian_backbone(s, t=t, gamma=gamma).toarray()
    return float(np.max(np.abs(Hg - Hh.astype(np.complex128))))


# ---------------------------------------------------------------------------
# Determinant phase via LU (argument principle); robust to over/underflow.
# ---------------------------------------------------------------------------
def _perm_parity(p):
    p = np.asarray(p); n = p.size
    seen = np.zeros(n, dtype=bool); cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1; j = i
            while not seen[j]:
                seen[j] = True; j = p[j]
    return (n - cycles) & 1


def det_phase(A, tries=5):
    """angle(det(A)) for sparse complex A via splu. Accumulates angles only (never the
    magnitude). Cost ~O(N) for near-banded A. If the contour energy lands exactly on an
    eigenvalue (singular factor, a measure-zero event), nudge the diagonal by a tiny
    complex jitter and retry - this shifts E off the spectrum generically and leaves the
    integer winding unchanged."""
    A = A.tocsc()
    N = A.shape[0]
    eps = 0.0
    for attempt in range(tries):
        try:
            B = A if eps == 0 else (A + eps * sp.identity(N, dtype=A.dtype, format="csc"))
            lu = spla.splu(B)   # default COLAMD + partial pivoting => robust to zero pivots
            ph = float(np.sum(np.angle(lu.U.diagonal())))
            if (_perm_parity(lu.perm_r) ^ _perm_parity(lu.perm_c)) & 1:
                ph += np.pi
            return ph
        except RuntimeError:
            eps = 1e-9 * (10 ** attempt) * (1.0 + 1.0j)
    raise RuntimeError("det_phase: matrix singular after %d jitter retries" % tries)


# ---------------------------------------------------------------------------
# INVARIANT 1: point-gap spectral winding W(E_ref) (the directionality invariant).
# ---------------------------------------------------------------------------
def winding(s, E_ref, g=0.6, t=1.0, gamma=6.0, n_phi=48):
    """W(E_ref) = (1/2pi) * net change of arg det(H(phi)-E_ref) as flux phi: 0->2pi.
    Non-Hermitian point-gap winding (argument principle). Clean HN ring: = sign(g) inside
    the spectral loop, 0 outside. Cost n_phi * O(N). Returns (rounded int, raw float)."""
    N = s.size
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    I = sp.identity(N, dtype=np.complex128, format="csc")
    angles = np.empty(n_phi, dtype=np.float64)
    for i, phi in enumerate(phis):
        angles[i] = det_phase(build_HN(s, g=g, t=t, gamma=gamma, bc="pbc", phi=phi) - E_ref * I)
    unwrapped = np.unwrap(np.concatenate([angles, angles[:1]]))
    w_raw = (unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi)
    return int(np.rint(w_raw)), float(w_raw)


# ---------------------------------------------------------------------------
# INVARIANT 2: well / skin mode position (Hermitian backbone density + skin envelope).
# ---------------------------------------------------------------------------
def well_mode_position(s, g=0.6, t=1.0, gamma=6.0, k_modes=8):
    """Deepest-well location from the Hermitian backbone low-energy density rho(x)=
    sum_{j<kk}|phi_j(x)|^2 (concentrates on the wells at {a, N-a}). argmax(rho) folded to
    [0,N/2) is the magnitude a. skin_com = e^{2gx}-weighted COM (the HN OBC right-eigvec
    density) = the imposed-directionality boundary pile-up. Returns
    (peak, peak_folded, ipr, skin_com). Robust: eigsh on the well-conditioned backbone."""
    N = s.size
    Hh = hermitian_backbone(s, t=t, gamma=gamma)
    v0 = np.random.default_rng(0).standard_normal(N)   # FIXED start => deterministic
    kk = min(k_modes, N - 2)
    vals, vecs = spla.eigsh(Hh, k=kk, which="SA", v0=v0, maxiter=8000, tol=1e-10)
    rho = np.sum(np.abs(vecs) ** 2, axis=1); rho = rho / np.sum(rho)
    peak = int(np.argmax(rho))
    peak_folded = min(peak, N - peak)
    ipr = float(np.sum(rho ** 2))
    xs = np.arange(N, dtype=np.float64)
    logw = 2.0 * g * xs + np.log(rho + 1e-300); logw -= logw.max()
    ws = np.exp(logw); ws = ws / ws.sum()
    skin_com = float(np.sum(xs * ws))
    return float(peak), float(peak_folded), ipr, skin_com


# ---------------------------------------------------------------------------
# The no-smuggle O(inst): the full non-Hermitian feature vector from PUBLIC data.
# ---------------------------------------------------------------------------
_G = 0.6
_T = 1.0
_GAMMA = 6.0
_N_PHI = 48
_E_REFS = (0.0, 0.6j, -0.6j, 1.5, -1.5, -2.0)


def O_hatano_nelson(inst):
    """Quadrature-synthesis candidate for the gate. Reads ONLY public k,b,N. Returns
    [ W(E)_raw for E in _E_REFS, peak/N, peak_folded/N, ipr, skin_com/N ]. Every entry is
    a function of the EVEN landscape and the FIXED constants g,gamma; never inst['d'] or
    the range. Gate's exact d-invariance audit -> delta == 0."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    s = score_landscape(k, b, N)
    feats = [winding(s, E, g=_G, t=_T, gamma=_GAMMA, n_phi=_N_PHI)[1] for E in _E_REFS]
    peak, peak_f, ipr, skin_com = well_mode_position(s, g=_G, t=_T, gamma=_GAMMA)
    feats.extend([peak / N, peak_f / N, ipr, skin_com / N])
    return np.asarray(feats, dtype=float)