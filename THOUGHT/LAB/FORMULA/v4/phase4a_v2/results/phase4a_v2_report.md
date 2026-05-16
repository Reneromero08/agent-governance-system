# Phase 4a Post-Mortem: What We Did vs What The Theory Required

**Date:** 2026-05-16
**Source:** Full Semiotic Light Cone 1.1 ingestion (8 documents)
**Reference:** `THOUGHT/LAB/FORMULA/v4/SEMIOTIC_LIGHT_CONE_1_1/`

---

## The Formula

```
R = (E / ∇S) × σ^Df
```

From the light cone, this is not a metaphor. It is an operational equation where every variable is measured from data, not assigned. The QEC domain validated it exhaustively (v1-v9, 100,000 shots per condition, train/holdout split, bootstrap CIs, alpha=0.70-0.73). The LLM domain has not yet been tested under the same methodology.

---

## What We Did (Phase 4a v1 + v2)

### v1: Static C + Temperature Modulation

1. Built C once from comprehension-time hidden states (model *reading* true/false claims)
2. Used C during token-by-token generation with T = 1/(R + ε)
3. Three conditions: CONTROL (T=0.7), CYBERNETIC (T=f(R)), VERIFY (T=0.7 + Lindblad)
4. 25 prompts across factual, reasoning, ambiguous, adversarial categories

**Result:** Loop neutral. With correct calibration (R_scale=500, T_base=3.0), accuracy reached parity with control (57.9% vs 63.2%). Without calibration (R_scale=100), accuracy collapsed to 18.8%.

### v2: Dynamic C + Context Feedback

1. C updated across runs using generation-time hidden states labeled by verification outcome
2. Context injection on verification failure — full chat reconstruction with correction message
3. 5 runs × 25 prompts

**Result:** Loop unstable. C diverged — successive Cs were nearly orthogonal (cos_sim < 0.02). Accuracy degraded from 57.9% (Run 1) to 29.4% (Run 5). Context corrections fired but were too vague to guide the model.

---

## What The Theory Actually Requires

### Mapping from QEC to LLM (from 01_FORMULA_5_2.md)

| Variable | QEC | LLM (correct) | What we did |
|----------|-----|---------------|-------------|
| **E** | Logical qubit (= 1.0) | Truth core (= 1.0) | Same |
| **∇S** | Physical error rate p | **Generation error rate per prompt category** | Never measured. Buried in C. |
| **σ** | Code compression ratio | **C separation power on generation states** | Confused with C matrix itself. Never varied. |
| **Df** | t = floor((d-1)/2) | **Number of verification/correction rounds** | Hardcoded to 7 (150/20). Never swept. |
| **R** | log_suppression = ln(p/pL) | **Truth amplification (accuracy gain)** | Measured as (h·w)² — C projection, not outcome. |

### The Kuramoto Condition (from 03_SEMIOTIC_WAVE_MECHANICS.md)

> Synchronization occurs when: **σ > ∇S**
> When symbolic compression exceeds the entropy gradient, meaning aligns spontaneously.

We never tested whether σ exceeds ∇S. The C we built separates true/false claims during *comprehension* (p≈0) but has no separation power during *generation* (σ ≈ 1). Below the Kuramoto threshold, the loop cannot synchronize — there is no phase transition. The model at T=0.7 is already near-optimal; the loop can only add noise.

### The QEC Methodology (from v9 code)

1. **Factorial parameter sweep** — Cartesian product of ∇S (p) × Df (d) levels
2. **Train/calibrate/holdout** — fit σ on training distances, predict on held-out distances
3. **σ varies with p** — not a single constant, measured per noise level
4. **E calibrated once** — median across ALL training points, should be ~1.0
5. **Bootstrap 95% CIs** — 1000 resamples on evaluation metrics
6. **No mid-circuit modification** — measure outcome, calibrate for NEXT run
7. **Explicit falsification** — list rejected hypotheses, not just successes

We did none of this.

---

## Gap Analysis

| Our approach | What the theory requires | Impact |
|---|---|---|
| C built from comprehension states | C built from **generation** states | C doesn't discriminate during generation (σ ≈ 1) |
| R = (h·w)^2 | R = accuracy gain over baseline | Measuring the wrong quantity |
| Dynamic C across runs | **Static C, sweep σ × Df** | Unstable feedback loop (echo chamber failure mode) |
| R_SCALE swept ad-hoc | σ^Df amplifier tested systematically | No evidence for exponential scaling |
| Context injection mid-generation | **Correct between runs** (QEC decoder pattern) | Mid-stream injection adds noise, not signal |
| Single C for all prompts | **σ varies per prompt category** (like QEC σ varies per p) | Masks category-dependent truth dynamics |
| Point estimates, no CIs | **Bootstrap 95% CIs** | Cannot distinguish signal from noise |
| "Loop is neutral" | **Explicit falsification list** | No clarity on what was rejected |

---

## What Phase 4a v3 Should Be (QEC-Aligned)

```
DESIGN:
  E = 1.0 (truth exists)
  ∇S = measured per prompt category (verification failure rate)
  σ = measured per category (C separation on generation states)
  Df = swept [1, 2, 3, 4, 5, 6, 7] (verification rounds)
  R = measured accuracy on held-out category × Df combination

SWEEP:
  ∇S levels: factual (low p), reasoning (med p), adversarial (high p) → 3 levels
  Df levels: 1..7 verification rounds → 7 levels
  → 21 conditions

TRAIN:
  Df ∈ {1, 2, 3}, factual + reasoning prompts (16 prompts)
  → Fit σ per ∇S level: ln(accuracy) = Df × ln(σ) + intercept
  → Calibrate E from training residuals

TEST:
  Df ∈ {4, 5, 6, 7}, adversarial prompts (9 prompts)
  → Predict: R_pred = (E/∇S) × σ^Df
  → Compare to actual accuracy
  → Bootstrap 95% CI on exponent

PREDICTIONS:
  1. σ > 1 for factual (C discriminates) → accuracy improves with Df
  2. σ ≈ 1 for adversarial (C cannot discriminate) → accuracy flat with Df
  3. The σ^Df functional form holds → ln(accuracy) linear in Df
  4. E ≈ 1.0 → formula structure confirmed

NO:
  - Dynamic C (unstable)
  - Context injection (adds noise)
  - Temperature modulation (not the independent variable)
  - Single global C (masks category effects)
```

---

## Files Created

### Phase 4a v1 (`THOUGHT/LAB/FORMULA/v4/phase4a/`)
- `phase4a_contrastive.py` — C builder from 40 contrastive claims (Fisher discriminant)
- `phase4a_loop.py` — Token-level control loop (3 conditions, calibrated T formula)
- `phase4a_analyze.py` — Honest statistical analysis with cross-run trends
- `phase4a_prompts.py` — 40 contrastive claims + 25 test prompts
- `phase4a_smoke.py` — Quick validation test
- `results/` — All condition data, C tensor, report

### Phase 4a v2 (`THOUGHT/LAB/FORMULA/v4/phase4a_v2/`)
- `phase4a_v2_loop.py` — Dynamic C + context feedback (5 runs)
- `phase4a_v2_smoke.py` — Smoke test validating both mechanisms
- `results/` — Per-run data, dynamic C snapshots

### Verification
- `LAW/CONTRACTS/_runs/REPORTS/phase4a/verification_report.md`

---

## Key Findings (For Future Work)

1. **The comprehension→generation gap is fundamental.** C built from reading doesn't transfer to writing. This is not a calibration issue — it's a representational mismatch. C must be built from generation-time hidden states.

2. **σ < ∇S everywhere we measured.** The Kuramoto condition for synchronization is not met at token granularity. Raw R ≈ 0.007 (1.7× random chance of 0.0004) is too weak to drive a phase transition. Step-level (multi-token window) aggregation may be required to cross the threshold.

3. **The σ^Df amplifier has never been tested.** We swept R_SCALE (an implementation parameter) but never tested whether accuracy scales exponentially with Df. This is the core prediction of the formula.

4. **Dynamic C is unstable without external grounding.** Each new C was nearly orthogonal to the previous (cos_sim < 0.02). The echo chamber failure mode predicted in 07_CYBERNETIC_TRUTH.md was confirmed: "C overfits to training biases; no external coupling."

5. **Mid-stream context injection is the wrong paradigm.** QEC decodes after the circuit runs, not during. The correction should happen between runs, not within a generation. The Lindblad operators should update C and T parameters for the next shot, not inject tokens into the current one.
