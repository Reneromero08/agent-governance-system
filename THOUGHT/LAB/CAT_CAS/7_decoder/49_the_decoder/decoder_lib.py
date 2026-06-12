"""
decoder_lib.py - shared engine for Exp 50 (The Decoder).

Reuses proven, in-lab holographic machinery (copied, not reinvented, because the
deprecated holo_core.py has a stale import path - the METHOD is live):

  - period_from_1d        : Exp 20 (20.10/6_shor_solver.py:57-66), torch FFT autocorrelation.
  - analyze_spectrum/project : Exp 34.8 working PCA fallback
                            (34_zeta_eigenbasis/34_2_holographic_sieves/8_riemann_harmonic_sieve.py:28-41).
  - phase_cavity_sieve    : live HOLO pipeline
                            (THOUGHT/LAB/HOLO/pipeline/02_cavity/fractal_cavity.py:48-55).

Plus the extractive frequency readout, the lookup-null decoder class, and stats.

THE EXTRACTIVE/LOOKUP SEPARATION (the heart):
  The answer (a global frequency / character) is recoverable only by COHERENT
  INTEGRATION over the full domain - SNR scales with integration length M.
  A decoder restricted to a bounded receptive field w << M (a "lookup-class"
  decoder) achieves SNR ~ A^2 w / sigma^2; the global spectral operator achieves
  A^2 M / sigma^2.  Choose amplitude A so A^2 w/sigma^2 < 1 < A^2 M/sigma^2 and the
  separation is structural - a locality barrier, NOT a compute handicap.  This is
  the abelian-HSP / Fourier-sampling advantage rendered as a measurement.
"""
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ===========================================================================
# Reused, proven holographic machinery (copied; see module docstring)
# ===========================================================================
def period_from_1d(signal):
    """Exp 20 (6_shor_solver.py:57-66). FFT autocorrelation -> argmax = period."""
    if not _HAS_TORCH:
        x = np.asarray(signal)
        if len(x) < 4:
            return 0
        spec = np.fft.fft(x)
        ac = np.fft.ifft(np.abs(spec) ** 2).real
        ac = ac / (ac[0] + 1e-15)
        sr = min(len(ac) // 2, 500000)
        if sr <= 2:
            return 0
        return int(np.argmax(np.abs(ac[2:sr]))) + 2
    sig = signal if torch.is_tensor(signal) else torch.tensor(np.asarray(signal))
    if len(sig) < 4:
        return 0
    spec = torch.fft.fft(sig)
    ac = torch.fft.ifft(torch.abs(spec) ** 2).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac) // 2, 500000)
    if sr <= 2:
        return 0
    _, mi = torch.max(torch.abs(ac[2:sr]), dim=0)
    return mi.item() + 2


def analyze_spectrum(obs):
    """Exp 34.8 working PCA fallback (8_riemann_harmonic_sieve.py:28-41).
    Hermitian covariance eigenvalues, descending."""
    cov = obs.T @ obs
    vals, _ = np.linalg.eigh(cov)
    return vals[::-1]


class _Proj:
    pass


def project(obs, fixed_k=10):
    """Exp 34.8 working PCA fallback. Top-k eigenbasis of the covariance."""
    cov = obs.T @ obs
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    p = _Proj()
    p.basis = vecs[:, :fixed_k].T
    p.eigenvalues = vals[idx]
    return p


def participation_dimension(eigvals):
    """D_pr = (sum lambda)^2 / sum(lambda^2) - effective rank."""
    e = np.abs(np.asarray(eigvals, dtype=float))
    s = e.sum()
    if s <= 0:
        return 0.0
    return float((s ** 2) / (np.sum(e ** 2) + 1e-30))


def phase_cavity_sieve(U, S, Vh, W_orig, keep_thresh=0.99):
    """Live HOLO pipeline (fractal_cavity.py:48-55). Greedy eigenmode sieve:
    discard a mode if reconstruction cosine-sim stays above keep_thresh."""
    if not _HAS_TORCH:
        raise RuntimeError("phase_cavity_sieve needs torch")
    def cosine_sim(Wo, Wr):
        X = torch.randn(20, Wo.shape[1])
        Yo = Wo.float() @ X.T
        Yr = Wr.float() @ X.T
        d = (Yo * Yr).sum(dim=0)
        return (d / (Yo.norm(dim=0) * Yr.norm(dim=0) + 1e-9)).mean().item()
    k = len(S)
    kept = list(range(k))
    for i in range(k - 1, -1, -1):
        keep = [j for j in kept if j != i]
        if not keep:
            continue
        Wt = (U[:, keep] * S[keep].unsqueeze(0)) @ Vh[keep, :]
        if cosine_sim(W_orig, Wt) > keep_thresh:
            kept.remove(i)
    return sorted(kept), [i for i in range(k) if i not in kept]


# ===========================================================================
# The extractive decoder: global coherent spectral readout
# ===========================================================================
def extract_fft_peak(E, meta=None):
    """Global FFT over the full domain -> dominant non-DC frequency bin.
    Coherent integration gain ~ len(E). This is the EXTRACTIVE operator."""
    x = np.asarray(E, dtype=np.complex128)
    M = len(x)
    spec = np.fft.fft(x)
    power = np.abs(spec) ** 2
    power[0] = 0.0
    half = M // 2
    return int(np.argmax(power[1:half]) + 1)


def extract_eigenmode_freq(E, meta=None, window=None):
    """FFT-independent extractive cross-check: build an observation matrix by
    sliding windows over the full signal, take the leading covariance eigenvector
    (analyze_spectrum/project), read its dominant frequency. Proves extraction is
    not specific to the FFT - any GLOBAL spectral operator recovers it."""
    x = np.asarray(E, dtype=np.complex128)
    M = len(x)
    L = window or max(8, M // 8)
    n_win = M - L + 1
    step = max(1, n_win // 256)
    rows = [np.concatenate([x[s:s + L].real, x[s:s + L].imag]) for s in range(0, n_win, step)]
    obs = np.asarray(rows)
    p = project(obs, fixed_k=2)
    v = p.basis[0]
    evec = v[:L] + 1j * v[L:]
    spec = np.abs(np.fft.fft(evec)) ** 2
    spec[0] = 0.0
    k_local = int(np.argmax(spec[1:L // 2]) + 1)
    return int(round(k_local * M / L))


# ===========================================================================
# The LOOKUP-NULL class: bounded receptive field OR bounded statistical order.
# Compute is UNBOUNDED; only the functional form is constrained.
# ===========================================================================
def _bin_to_full(k_local, w, M):
    return int(round(k_local * M / w))


def null_windowed_fft(E, meta=None, window_frac=1.0 / 16):
    """N3: best local periodogram over a window w << M. Receptive field w cannot
    achieve the coherent gain of the full domain."""
    x = np.asarray(E, dtype=np.complex128)
    M = len(x)
    w = max(8, int(M * window_frac))
    seg = x[:w]
    spec = np.abs(np.fft.fft(seg)) ** 2
    spec[0] = 0.0
    k_local = int(np.argmax(spec[1:w // 2]) + 1)
    return _bin_to_full(k_local, w, M)


def null_windowed_kay(E, meta=None, window_frac=1.0 / 16):
    """N2: local lag-1 phase (Kay) frequency estimator over a window w << M.
    A global statistic of a SHORT window - same locality barrier."""
    x = np.asarray(E, dtype=np.complex128)
    M = len(x)
    w = max(8, int(M * window_frac))
    seg = x[:w]
    z = np.sum(seg[:-1] * np.conj(seg[1:]))
    ang = np.angle(z)            # ~ -2 pi k / M  (per sample step)
    k = (-ang) / (2 * np.pi) * M
    return int(round(k)) % (M // 2)


def null_windowed_autocorr(E, meta=None, window_frac=1.0 / 16):
    """N1: local autocorrelation period over a window w << M, mapped to a
    frequency. Bounded receptive field."""
    x = np.asarray(E, dtype=np.complex128)
    M = len(x)
    w = max(8, int(M * window_frac))
    r = period_from_1d(x[:w])
    if r <= 1:
        return 0
    return int(round(M / r))


def null_histogram_regressor(E, meta=None, train=None, bins=24):
    """N4: estimate the frequency from a value HISTOGRAM + low-order moments via a
    nearest-neighbour regressor trained on labelled examples. Bounded statistical
    order: the histogram of A*e^{i theta}+noise is ~rotation-invariant, so it is
    (asymptotically) independent of the frequency. Provably f-blind -> chance.
    `train` = list of (E_i, k_i)."""
    def feats(sig):
        s = np.asarray(sig, dtype=np.complex128)
        mag = np.abs(s)
        h, _ = np.histogram(np.angle(s), bins=bins, range=(-np.pi, np.pi), density=True)
        return np.concatenate([[mag.mean(), mag.std(), np.abs(s.mean())], h])
    f_test = feats(E)
    if not train:
        return 0
    Xtr = np.asarray([feats(e) for e, _ in train])
    ytr = np.asarray([k for _, k in train])
    d = np.linalg.norm(Xtr - f_test[None, :], axis=1)
    kk = min(3, len(ytr))
    nn = np.argsort(d)[:kk]
    return int(round(np.mean(ytr[nn])))


# Registries (the word "null" appears here; orchestrators import these).
EXTRACTIVE_DECODERS = {
    "fft_peak": extract_fft_peak,
    "eigenmode": extract_eigenmode_freq,
}
LOOKUP_NULLS = {
    "null_windowed_fft": null_windowed_fft,
    "null_windowed_kay": null_windowed_kay,
    "null_windowed_autocorr": null_windowed_autocorr,
    "null_histogram_regressor": null_histogram_regressor,
}


def null_model(name):
    """Return a lookup-null decoder by name (shared null harness)."""
    return LOOKUP_NULLS[name]


# ===========================================================================
# Statistics (numpy-only; no scipy dependency)
# ===========================================================================
def wilson_ci(successes, n, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_h(p1, p2):
    """Effect size for two proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
    phi2 = 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))
    return float(phi1 - phi2)


def cohen_d(a, b):
    """Standardized mean difference."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-30)
    return float((a.mean() - b.mean()) / (sp + 1e-30))


def bootstrap_ci(samples, ci=0.95, n_boot=2000, seed=0):
    """Percentile bootstrap CI of the mean."""
    s = np.asarray(samples, float)
    if len(s) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot = rng.choice(s, size=(n_boot, len(s)), replace=True).mean(axis=1)
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def permutation_p(success_a, success_b, n_perm=10000, seed=0):
    """One-sided permutation p-value that group A's success rate exceeds B's.
    Inputs are 0/1 arrays (per-trial success indicators)."""
    a = np.asarray(success_a, float); b = np.asarray(success_b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if (pool[:na].mean() - pool[na:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)
