# Exp 50 L4 Gate Design Audit

**Date:** 2026-06-14.
**Status:** L4 BLOCKED. No code attempted.

---

## Executive Verdict

L4 implementation was not attempted. The naive map `f(x) = x if verify(x) else (x+1) mod N`
is rejected for three independent reasons:

1. **Forward scan.** It enumerates x sequentially. Wrapping it in SHA-verified tape
   lifecycle is ceremonial, not catalytic.

2. **Fold-even verify.** `verify(x) = score(x) > M/4` where `score(x) = sum b_i cos(2*pi*k_i*x/N)`.
   Since cosine is even, `verify(d) == verify(N-d)` always. The accepting set is the full
   orbit {d, N-d}. The scan restricted to [1, N/2) finds `a = min(d, N-d)`, not `d`.

3. **Target collapse.** The charter incorrectly restricts `d` to `[1, N/2)`. Under that
   restriction, `d = a` (the public fold magnitude), and "recovering d" is recovering the
   candidate value -- a problem solved at AUC 1.000 by public readout. The actual
   construction (`construction.py` line 71-77) draws `d` from the full `[1, N)`.

**Correct target:** Full `d` recovery over the full fold pair domain requires orientation
recovery `1[d < N/2]`. Phase 6 proved this is information-theoretically absent from the
public cosine channel under no-smuggle constraints.

**L4 is BLOCKED on public verify(x) alone.** A genuinely new substrate mechanism --
not sequential C code wrapped in SHA tape lifecycle -- is required.

---

## Current Ladder State

| Gate | Status | Commit | Key Result |
|---|---|---|---|
| L2 | PASS | `db852286` | Tape lifecycle: SHA restore, 50/50, 4 controls |
| L3 | PASS | `c394e30b` | Contraction map convergence, 90/90, 2-9 iterations |
| L4 | **BLOCKED** | -- | Forward scan + fold symmetry + target collapse |
