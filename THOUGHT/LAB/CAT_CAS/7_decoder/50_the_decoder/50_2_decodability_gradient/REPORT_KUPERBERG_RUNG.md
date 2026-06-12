# Exp 50.2d - The Kuperberg Rung: the dihedral barrier is subexponential, not polynomial

**Verdict:** `DIHEDRAL_BARRIER_SUBEXPONENTIAL_UPPER_BOUND` (all 4 gates pass, entry exits 0).
**Claim level:** 4 (a measured subexponential upper bound from a real sieve simulation; the
2^{O(sqrt n)} rate and the super-polynomial lower bound are cited from Regev/Kuperberg +
50.2c, not re-proven here).

## The gap this fills

50.2c sandwiched the dihedral-HSP slope between info-cheap (O(sqrt N) coset states
determine d) and compute-hard (a poly(n)-budget secret search fails; full recovery is
O(2^n)). It left the **middle rung** unmeasured. Kuperberg's collimation sieve recovers the
slope in 2^{O(sqrt n)} queries - subexponential, below the 2^n full search and above the
poly(n) budget that fails. This brick builds that sieve and measures its query cost, pinning
the barrier as a sandwich:

```
poly(n) [FAILS, 50.2c lower bound]  <  2^{O(sqrt n)} [this sieve, upper bound]  <  2^n
```

## Mechanism

A dihedral coset state carries a label k in [0,N), N=2^n, with phase e^{2 pi i k d / N}. The
sieve is **d-agnostic** - it only manipulates labels. Combining two states sharing their low
b bits, |k_i> and |k_0>, yields |k_i - k_0> whose low b bits are zero. Clearing b bits per
round for ~n/b rounds produces a state with label 2^{n-1}, whose phase is e^{i pi d} =
(-1)^d - one secret bit, read by measurement. b = round(sqrt(n)) balances states-lost-per-
round against rounds, giving 2^{O(sqrt n)} total queries. No catalytic tape (this measures an
algorithm's query scaling; a tape would be ceremonial).

## Results (n = 6..30, N up to ~1.07e9)

| n | b | M_needed | log2(M) | 2^n / M_needed | production |
|---|---|---|---|---|---|
| 6 | 2 | 16 | 4.0 | 4 | 1.00 |
| 14 | 4 | 64 | 6.0 | 256 | 1.00 |
| 22 | 5 | 128 | 7.0 | 3.3e4 | 1.00 |
| 30 | 5 | 256 | 8.0 | 4.2e6 | 1.00 |

- **Readout correctness:** conditional correctness P(bit correct | useful state produced) =
  **1.000** at every tested n (the `(-1)^d` readout is exact). Phase-randomised null = 0.44-
  0.51 (chance), so the sieve's success is the real coset phase, not an artifact.
- **Subexponential:** 2^n / M_needed grows from **4x to 4.2 million x** over n=6..30; log2(M)
  reaches only 8 at n=30 (log2(M)/n = 0.27, clearly sublinear).
- **Rate:** log2(M_needed) ~ a*sqrt(n) fit R^2 = **0.948** (a=1.33), consistent with
  Kuperberg 2^{O(sqrt n)}.

## Gates

| Gate | Result | Detail |
|---|---|---|
| G1 readout correctness (bit \| produced) ~ 1.0 | PASS | min conditional correctness = 1.000 |
| G2 subexponential: M_needed << 2^n, gap widens >10x | PASS | gap 4x -> 4.2e6x; log2(M)=8 << n=30 |
| G3 subexponential rate (sublinear in n; sqrt-consistent) | PASS | sqrt-fit R^2=0.948; log2(M)/n=0.27 |
| G4 phase-randomised null reads bit only at chance | PASS | null correctness = 0.483 |

## Honest scope (what is NOT claimed)

The super-polynomial side is **not** claimed from the rate fit. Over the reachable n range a
2^{sqrt n} curve and a low-degree polynomial are statistically indistinguishable: sqrt-fit
R^2=0.948 vs poly-fit R^2=0.928. We report both and decline to separate them empirically.
Super-polynomiality is inherited from 50.2c's poly(n)-budget failure (success -> 0 as N
grows) plus the standard Regev/Kuperberg result, which we cite. This brick contributes only
the **subexponential upper bound** (the sieve works, decisively below 2^n) and the verified
readout.

## Why it matters

This completes the characterization of the bedrock the whole holographic decoder bottoms out
on: the dihedral / lattice (unique-SVP) wall is **super-polynomial but subexponential** - no
poly(n) readout exists, but it is not full-exponential either. That is the precise shape of
the barrier 50.4 showed Exp 25's holographic LWE attack fails to cross.

## Reproduce

```
python THOUGHT/LAB/CAT_CAS/7_decoder/50_the_decoder/50_2_decodability_gradient/50_2d_kuperberg_sieve.py
```
Writes `kuperberg_result.json` + `output_kuperberg.txt`. Exits 0 iff all gates pass.
