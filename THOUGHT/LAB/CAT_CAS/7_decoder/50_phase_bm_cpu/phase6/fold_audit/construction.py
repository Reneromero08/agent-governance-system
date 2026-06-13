"""
construction.py - the REAL Exp 50.14 public fixed-point map, lifted verbatim from
THOUGHT/LAB/CAT_CAS/49_the_decoder/49_14_reversible_substrate/49_14_substrate.py
(functions coset_samples / make_verify) and the SPEC_PHASE6 Section 2 definition.

This module is the single source of truth for the construction used by the Stage 1
fold audit. It introduces NO new modeling choices: the parametric form, the noise
model on b, the sample count M, the score, and the [1, N/2) pinning of d are all
copied from the lab's actual brick. ASCII only; all RNGs are seeded by the caller.

Construction (Exp 50.14 / SPEC_PHASE6 Sec 2), parametrized by n with N = 2^n:

  Public sampling (coset_samples, verbatim):
      k_i ~ Uniform{0, ..., N-1},  i = 1..M
      p_i = (1 + cos(2*pi*k_i*d/N)) / 2
      b_i = +1 with prob p_i else -1        =>   E[b_i] = cos(2*pi*k_i*d/N)

  Public score / verifier (make_verify, verbatim):
      score(x)  = sum_i b_i * cos(2*pi*k_i*x/N)        # O(M); uses (k,b) only, never d
      accept(x) = score(x) > M/4                       # true iff x in {d, N-d}
      f(x)      = x if accept(x) else (x+1) mod N      # unique fixed point in [1,N/2)

  Sample count (verbatim from the brick):
      M = max(4 * ceil(sqrt(N)), 48 * n)               # M ~ O(sqrt N) public samples

  Secret pinning:
      d is drawn in [1, N), and the fixed point lives at min(d, N-d) in [1, N/2).
      The orientation bit that the fold sigma: d -> N-d destroys is
          b_orient = 1[d < N/2]
      (i.e. is the true d on the lower or upper half of the circle).
"""
import numpy as np


def M_for(n):
    """Sample count M, verbatim from 49_14_substrate.py main()."""
    N = 1 << n
    return int(max(4 * int(np.ceil(np.sqrt(N))), 48 * n))


def coset_samples(N, d, M, rng):
    """Verbatim from 49_14_substrate.py: the lab's real public-data generator.
    Returns (k, b) with k ~ Uniform[0,N), b in {-1,+1}, E[b]=cos(2 pi k d / N)."""
    k = rng.integers(0, N, size=M)
    p = (1 + np.cos(2 * np.pi * k * d / N)) / 2
    b = np.where(rng.random(M) < p, 1.0, -1.0)
    return k, b


def score(k, b, x, N):
    """Public matched-filter score(x) = sum_i b_i cos(2 pi k_i x / N). O(M), no d."""
    return float(np.dot(b, np.cos(2 * np.pi * k * x / N)))


def make_verify(k, b, N):
    """Verbatim verifier from the brick: accept iff score(x) > M/4."""
    M = len(b)
    thresh = M / 4.0

    def verify(x):
        return score(k, b, x, N) > thresh

    return verify, M


def orientation_bit(d, N):
    """The bit the fold destroys: 1 if the true secret is on the lower half."""
    return int((d % N) < (N / 2))


def sample_secret(N, rng):
    """Draw a hidden secret d in [1, N), excluding the fold-fixed points 0 and N/2
    (where d == N-d so orientation is undefined). Returns d in {1,...,N-1}\\{N/2}."""
    while True:
        d = int(rng.integers(1, N))
        if d != N // 2:               # exclude the self-paired Nyquist point
            return d


def fold(d, N):
    """The construction's symmetry sigma: d -> N-d."""
    return (N - d) % N


# ---------------------------------------------------------------------------
# Public channels: what an auditor is allowed to read.
# ---------------------------------------------------------------------------
def cosine_channel(k, d, N):
    """The REAL (even) channel the public data lives in: c_k = cos(2 pi k d / N).
    This is the noiseless mean of the public bits; it is sigma-invariant."""
    return np.cos(2 * np.pi * k * d / N)


def quadrature_channel(k, d, N):
    """The complex coefficient z_k = exp(-2 pi i k d / N). Its imaginary part is
    -sin(2 pi k d / N), the ODD channel that is ABSENT from the public cosine data.
    Reading this is the crossing the SPEC describes; it is NOT public."""
    return np.exp(-2j * np.pi * k * d / N)


def dyadic_ladder(n):
    """The frequency ladder k = N/2, N/4, ..., 2, 1 used for one-shot phase
    estimation (SPEC 1C CROSSING SPEC). N = 2^n, so these are exact powers of two."""
    N = 1 << n
    return np.array([N >> j for j in range(1, n + 1)], dtype=np.int64)  # N/2 ... 1
