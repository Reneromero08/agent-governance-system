# Report: What Q1 (Why `∇S`?) Being “Solved” Actually Means

**Question:** `THOUGHT/LAB/FORMULA/research/questions/critical/q01_why_grad_s.md`  
**Status in system:** **ANSWERED** (with explicit scope/assumptions)  

---

## Executive claim (scoped)
Q1 is “solved” in the following precise sense:

> If your resonance is built from a **location–scale** uncertainty model, then dividing by a local dispersion/scale term (`∇S`, “std-like”) is not a stylistic choice — it is the **normalization factor** that turns a bounded compatibility score into a **comparable evidence density** across scales.

This does **not** claim the full formula is complete; it pins down why the denominator exists and what it must represent.

---

## The core mathematical move: make error dimensionless
If error has units (meters, tokens, logits, etc.), it is meaningless to compare raw error across contexts with different scale.

So you define the dimensionless residual:

`z := error / s`

where `s` is a local scale parameter (what `∇S` is estimating).

Once you do this, any “shape”/compatibility term must be a function of `z`, not raw error.

---

## Why division is forced (location–scale families)
Any location–scale likelihood has the form:

`p(x | μ, s) = (1/s) f((x-μ)/s)`

Evaluate at the target/“truth” point and define `E(z) := f(z)`:

`p(truth | μ, s) = const · E(z) / s`

So the denominator is simply the `1/s` normalization that every location–scale family carries.

**Interpretation in your language:**
- `E(z)` = “compatibility with reality” (shape term, bounded)
- `∇S ≈ s` = “how wide the local uncertainty is” (scale)
- `E/∇S` = “how much compatible evidence per unit uncertainty” (evidence density)

---

## What the Free Energy link becomes (and what it does NOT claim)
If you choose the Gaussian family:

- `E(z) = exp(-z^2/2)`
- `R = E(z)/s`

then the Gaussian variational free energy has the form:

`F = z^2/2 + log(s) + const`

and you get an exact identity:

`log(R) = -F + const`  
`R ∝ exp(-F)`

**Meaning:** in that specified family, maximizing `R` is equivalent (up to a constant) to minimizing free energy.

**What this does not claim:** that *every* empirical domain will obey the Gaussian mapping, or that the full formula’s other term (`σ^Df`) is derived from FEP.

---

## Why `std` vs `MAD` stops being a fight
Q1 also resolves a recurring confusion:

> The denominator is “the scale parameter,” not “standard deviation specifically.”

- Gaussian noise ⇒ scale parameter is `std`
- Laplace noise ⇒ scale parameter is `b` (MAD-like)

So “std vs MAD” is not an argument about math correctness; it’s a **model choice about tails / loss geometry** (L2 vs L1).

---

## What Q1 closes vs what stays open

### Closed by Q1 (hard)
- Why a dispersion term must be in the denominator at all: it is the scale normalization of any location–scale family.
- Why the residual must be dimensionless (`z = error/scale`) if you want cross-context comparability.
- The cleanest analytic bridge to FEP in a specified family (Gaussian): `log(R) = -F + const`.

### Still open (explicitly not covered by Q1)
- The principled derivation/interpretation of `σ^Df` in the full formula `R = (E/∇S) × σ^Df`.
- Which likelihood family you should assume in any given domain (Gaussian vs Laplace vs heavier tails).
- How to define `∇S` when the “neighborhood” is not a simple metric ball (e.g., graph actions, symbolic manifolds, non-Euclidean contexts).

---

## Where the proof lives
- Narrative proof + scope notes: `THOUGHT/LAB/FORMULA/research/questions/critical/q01_why_grad_s.md`
- Mechanical proof of the exact Gaussian identity: `THOUGHT/LAB/FORMULA/experiments/open_questions/q1/q1_derivation_test.py`

