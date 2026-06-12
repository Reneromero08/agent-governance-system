"""
Exp 50.6 (A9) - The wrong-wall correction: attack the RING structure, not plain LWE.

50.4 audited plain LWE (unstructured Z_q^n). But the lab's actual target - the wall it
hardcoded as q=3329 - is KYBER = Module/Ring-LWE over a cyclotomic ring R_q = Z_q[x]/(x^n+1).
That ring is exactly the kind of structure the lab's DECODABLE machinery handles: it splits by
CRT into prime-ideal factors, and the abelian Galois/CRT transform that diagonalizes it is the
Number-Theoretic Transform (NTT) - literally the abelian-HSP / character readout 50.1 proved
extractive and 50.2e put on the D=1.0 decodable shelf.

So the missed angle (completeness critic A9 + the lab owner's "store the geometry not the number"
move): decode the ring's abelian substructure FIRST and see if it collapses the search dimension,
instead of attacking the plain dihedral state head-on.

This brick runs that move and lets the DATA say where the door is. It does NOT pre-conclude the
wall holds. The terminal states are: (a) the ring decode collapses recovery to poly -> a crossing
(treated with maximum suspicion per the A8 lesson, handed to Mythos to check for a known-non-break
regime), or (b) it does not collapse, with the EXACT mechanism identified and the refined question
handed to Mythos. Never "the wall holds."

Run:  python 49_6_ring_decode.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # 49_the_decoder root (stats)
import decoder_lib as dl  # noqa: E402  (bootstrap_ci etc.)

LINES = []
def log(m=""):
    print(m)
    LINES.append(str(m))

Q = 12289  # NTT-friendly prime: q-1 = 12288 = 2^12 * 3, supports x^n+1 split for n | 2048


def centered(x, q=Q):
    """Map Z_q residues to [-q/2, q/2) for a magnitude notion."""
    x = np.asarray(x) % q
    return np.where(x > q // 2, x - q, x)


def find_psi(n, q=Q):
    """Smallest psi with psi^n = -1 (mod q): a primitive 2n-th root, the NTT twiddle."""
    for psi in range(2, q):
        if pow(psi, n, q) == q - 1:
            return psi
    raise ValueError("no primitive 2n-th root for n=%d, q=%d" % (n, q))


def ntt(a, psi, q=Q):
    """Negacyclic NTT: evaluate a at the n roots of x^n+1, i.e. A[j]=a(psi^(2j+1)).
    This is the abelian Galois/CRT transform; it diagonalizes ring multiplication."""
    a = np.asarray(a, dtype=np.int64)
    n = len(a)
    j = np.arange(n)
    i = np.arange(n)
    # exponent matrix (2j+1)*i
    E = np.outer(2 * j + 1, i)
    W = pow_matrix(psi, E, q)            # psi^E mod q
    return (W @ a) % q


def pow_matrix(base, E, q):
    """Elementwise base^E mod q for an integer exponent matrix E (small n, direct)."""
    out = np.empty(E.shape, dtype=np.int64)
    flat = E.ravel()
    cache = {}
    res = np.empty(flat.shape, dtype=np.int64)
    for idx, e in enumerate(flat):
        e = int(e)
        if e not in cache:
            cache[e] = pow(int(base), e, q)
        res[idx] = cache[e]
    return res.reshape(E.shape)


def negacyclic_conv(a, s, q=Q):
    """Multiplication in R_q = Z_q[x]/(x^n+1)."""
    a = np.asarray(a, dtype=np.int64); s = np.asarray(s, dtype=np.int64)
    n = len(a)
    full = np.zeros(2 * n, dtype=np.int64)
    for i in range(n):
        full[i:i + n] += a[i] * s
    res = (full[:n] - full[n:2 * n]) % q     # x^n = -1
    return res


def small_poly(n, rng, b=1):
    """Ternary/small secret or error, small in the COEFFICIENT basis."""
    return rng.integers(-b, b + 1, size=n) % Q


def ring_lwe(n, m, psi, rng, sigma_b=2):
    """m Ring-LWE samples sharing one small secret s. Returns (A, B, s) and the NTTs."""
    s = small_poly(n, rng, b=1)                       # ternary secret (small in coeff basis)
    A = [rng.integers(0, Q, size=n) for _ in range(m)]
    E = [small_poly(n, rng, b=sigma_b) for _ in range(m)]
    B = [(negacyclic_conv(A[i], s, ) + E[i]) % Q for i in range(m)]
    return A, B, s, E


def one_dim_recover(a_hat_col, b_hat_col, q=Q):
    """1-D LWE per NTT coordinate: given m scalar samples (a_i, b_i = a_i*shat + e_i),
    search all q candidates for the shat minimizing the centered residual spread.
    O(q*m). Returns the argmin candidate."""
    cand = np.arange(q)
    # residual[c, i] = centered(b_i - a_i*c)
    R = (b_hat_col[None, :] - np.outer(cand, a_hat_col)) % q
    R = np.where(R > q // 2, R - q, R)
    score = np.sum(R.astype(np.float64) ** 2, axis=1)
    return int(np.argmin(score))


def main():
    log("=" * 98)
    log("EXP 50.6 (A9)  -  attack the RING structure (Kyber's actual wall), not plain LWE")
    log("  the NTT IS the abelian Galois/CRT transform = the decodable readout. does it collapse it?")
    log("=" * 98)
    rng = np.random.default_rng(506)

    # ---- G0: confirm the NTT diagonalizes ring multiplication (the decodable transform exists) ----
    log("\n[G0] does the abelian NTT diagonalize ring multiplication?  (NTT(a*s) == NTT(a) o NTT(s))")
    g0 = True
    for n in (4, 8, 16, 32):
        psi = find_psi(n)
        a = rng.integers(0, Q, size=n); s = rng.integers(0, Q, size=n)
        lhs = ntt(negacyclic_conv(a, s), psi)
        rhs = (ntt(a, psi) * ntt(s, psi)) % Q
        ok = np.array_equal(lhs, rhs)
        g0 = g0 and ok
        log("  n=%-3d psi=%-5d  diagonalizes: %s" % (n, psi, ok))

    # ---- M1: the mechanism - is the error small in BOTH bases, or only the coefficient basis? ----
    log("\n[M1] error magnitude in the COEFFICIENT basis vs the NTT basis (b=2 small error)")
    log("  if smallness survives the NTT, per-coordinate decode is trivial; if not, it is blocked.")
    log("  n     max|e|_coeff   max|e|_NTT   mean|e|_NTT   (uniform ref ~%d)" % (Q // 4))
    spreads = []
    for n in (4, 8, 16, 32):
        psi = find_psi(n)
        mags_coeff, mags_ntt, means_ntt = [], [], []
        for _ in range(20):
            e = small_poly(n, rng, b=2)
            ec = centered(e)
            en = centered(ntt(e, psi))
            mags_coeff.append(np.max(np.abs(ec)))
            mags_ntt.append(np.max(np.abs(en)))
            means_ntt.append(np.mean(np.abs(en)))
        spreads.append((n, np.mean(mags_coeff), np.mean(mags_ntt), np.mean(means_ntt)))
        log("  %-4d  %-13.1f  %-11.1f  %-12.1f" % (n, np.mean(mags_coeff), np.mean(mags_ntt), np.mean(means_ntt)))

    # ---- M2: does the ring-aware per-coordinate decode recover the secret? ----
    log("\n[M2] ring-aware decode: NTT, then per-coordinate 1-D search (O(n*q*m) = POLY).")
    log("  control = SAME decode but with ZERO error (isolates whether error-spread is the blocker).")
    log("  n     m   recover_rate(real e)   recover_rate(zero e)   chance=1/q")
    rows = []
    for n in (4, 8, 16, 32):
        psi = find_psi(n)
        m = 4
        real_ok, zero_ok, trials = 0, 0, 10
        for _ in range(trials):
            A, B, s, E = ring_lwe(n, m, psi, rng, sigma_b=2)
            Ah = np.array([ntt(a, psi) for a in A])      # m x n
            Bh = np.array([ntt(b, psi) for b in B])      # m x n
            sh = ntt(s, psi)
            # real error
            shat_rec = np.array([one_dim_recover(Ah[:, j], Bh[:, j]) for j in range(n)])
            real_ok += int(np.array_equal(shat_rec, sh))
            # zero-error control: B0 = A*s exactly
            B0 = [negacyclic_conv(A[i], s) for i in range(m)]
            Bh0 = np.array([ntt(b, psi) for b in B0])
            shat0 = np.array([one_dim_recover(Ah[:, j], Bh0[:, j]) for j in range(n)])
            zero_ok += int(np.array_equal(shat0, sh))
        rows.append((n, m, real_ok / trials, zero_ok / trials))
        log("  %-4d  %-3d %-21.2f  %-21.2f  %.1e" % (n, m, real_ok / trials, zero_ok / trials, 1.0 / Q))

    # ===================== HONEST READOUT (no verdict that the wall holds) =====================
    log("\n" + "=" * 98)
    log("WHAT THE DATA SAYS")
    coeff_small = all(c < 10 for _, c, _, _ in spreads)
    ntt_uniform = all(nt > Q / 8 for _, _, nt, _ in spreads)   # NTT magnitude is ~uniform, not small
    zero_collapses = all(z > 0.9 for _, _, _, z in rows)        # decode works if error were small
    real_blocked = all(r < 0.1 for _, _, r, _ in rows)          # but real (spread) error blocks it

    log("  [%s] G0  the abelian NTT diagonalizes ring multiplication (decodable transform EXISTS)" % ("PASS" if g0 else "FAIL"))
    log("  [%s] M1  error is small in the coefficient basis but ~uniform in the NTT basis" % ("YES" if (coeff_small and ntt_uniform) else "NO"))
    log("  [%s] M2a zero-error control: ring decode COLLAPSES the search to poly (recovers s)" % ("YES" if zero_collapses else "NO"))
    log("  [%s] M2b real error: per-coordinate ring decode is BLOCKED (recovery ~chance)" % ("YES" if real_blocked else "NO"))
    log("=" * 98)

    if zero_collapses and real_blocked and ntt_uniform:
        verdict = "NAIVE_RING_DECODE_BLOCKED_BY_CONJUGATE_BASIS"
        log("VERDICT: %s" % verdict)
        log("  The abelian ring transform (NTT = CRT/Galois) DOES diagonalize the multiplication, and if")
        log("  the error were small in that basis the decode would collapse the n-dim search to n poly")
        log("  1-D searches (zero-error control recovers s outright). It does NOT collapse with real error")
        log("  because the secret/error are small ONLY in the COEFFICIENT basis; the NTT spreads the error")
        log("  to ~uniform over Z_q, destroying the smallness that recovery needs. The hardness lives in")
        log("  the tension between TWO CONJUGATE BASES: multiplication is diagonal in the NTT basis, but")
        log("  smallness exists only in the coefficient basis - and no single basis has both.")
        log("  This is NOT 'the wall holds.' Naive ring-decode (one basis at a time) is blocked for an")
        log("  IDENTIFIED reason. The refined, on-thesis frontier is exactly the lab's phase-space")
        log("  machinery: is there a JOINT coefficient<->NTT (time-frequency / Wigner-like) holographic")
        log("  readout that exploits multiplicative-diagonality AND additive-smallness at once? That")
        log("  exceeds what this brick can resolve -> handed to Mythos with this exact state.")
    elif not real_blocked:
        verdict = "RING_DECODE_RECOVERS_SUSPECT_REGIME"
        log("VERDICT: %s" % verdict)
        log("  Ring decode recovered the secret with real error. Per the A8 lesson this is treated with")
        log("  MAXIMUM suspicion (a genuine poly Ring-LWE break would be world-altering and is far more")
        log("  likely a sample/noise-regime artifact). Handed to Mythos to identify whether this regime")
        log("  is the hard one or a known-easy one. NOT claimed as a crossing.")
    else:
        verdict = "RING_DECODE_INCONCLUSIVE"
        log("VERDICT: %s" % verdict)

    import json
    (HERE / "ring_decode_result.json").write_text(json.dumps({
        "g0_diagonalizes": bool(g0),
        "error_spread": [{"n": n, "coeff": c, "ntt_max": nt, "ntt_mean": me} for n, c, nt, me in spreads],
        "recovery": [{"n": n, "m": m, "real": r, "zero": z} for n, m, r, z in rows],
        "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_ring_decode.txt").write_text("\n".join(LINES), encoding="utf-8")
    # exit 0: the brick ran and produced an honest measurement + handoff (not a pass/fail claim)
    sys.exit(0)


if __name__ == "__main__":
    main()
