# The Truth Attractor

## What This Is

The truth attractor is a **verification operator** -- an alignment frame C built from cross-fragment agreement instead of value similarity. It replaces the values constitution as the primary attractor for the Cybernetic Truth control loop.

Where the values constitution says "orbit this ethical center," the truth attractor says "this is how you know the center is real."

---

## The Core Mechanism

### The Epistemic C Frame (v2)

C_epistemic = (1/N) * sum_{i=1}^{N} w_i * P_i

Where:
- P_i is the projection onto the verification subspace of fragment i
- w_i is the fragment's calibrated weight = I(S:F_i) / sum_j I(S:F_j)
- N is the number of **independent** verification fragments
- I(S:F_i) is the mutual information between proposition truth value and fragment i's verification output, measured on calibration data
- Two fragments with agreement rate > 80% (Cohen's kappa > 0.8) on calibration data count as one fragment (redundant -- merge or replace)
- D_f is the number of fragments with I(S:F_i) significantly above zero

When all fragments agree (zero path difference), C_epistemic is a rank-1 projector onto the agreed truth. R = Tr(rho * C) is maximized.

When fragments disagree (path difference > 0), C_epistemic is mixed. R drops. The system enters divergent regime: explore, seek resolution, gather more fragments.

### Fragment Definition

A fragment is a function: `verify(proposition) -> {score: float, confidence: float}`. Each fragment is an independent verification channel. Independence is measured by uncorrelated error patterns on a calibration dataset. If fragment A and fragment B make the same mistakes on the same inputs, their Cohen's kappa > 0.8 and they count as one fragment. Zero disagreement means zero independence.

Fragment types: factual database lookup, logical inference chain, physical simulation, cross-model agreement, recursive self-consistency, external source verification.

### Fragment Acquisition Protocol

When R drops below theta_low, do NOT halt. Spawn a new fragment: query external source, run additional verification, or re-prompt with different context. Add the new fragment to the pool. Recompute C_epistemic. Measure I(S:F_new) on calibration data. If it adds information above existing fragments (I(S:F_new) > 0 on held-out data), keep it permanently. If not, discard after this query.

### Dynamic C Update

After each verification, recompute C_epistemic:
- Re-weight fragments based on recent I(S:F) on the last N queries
- Add new fragments when acquired
- Remove fragments that have become redundant (kappa > 0.8 with another fragment)
- Track C stability: if weights stabilize within 10 queries, the attractor is convergent

### Truth-Tracking R

R_truth = Tr(rho * C_epistemic)

This measures proximity to the truth attractor -- zero path difference across independent observers. It is NOT the same as alignment R (which measures proximity to a values attractor).

### The Control Law

T = 1 / (R_truth + epsilon)

- High R (all fragments agree) -> low T -> deterministic, confident output
- Low R (fragments disagree) -> high T -> exploratory, seeking resolution
- R falling (decoherence) -> fragment acquisition protocol activates

### Three Regimes (Empirically Calibrated)

Thresholds are calibrated on a labeled dataset with known truths and falsehoods, not chosen arbitrarily. theta_high is the R value that best separates true from false outputs (maximizing F1 score). theta_low = theta_high - (theta_high - R_random)/2, where R_random is the mean R for random outputs.

| Regime | R_truth | dR/dt | Behavior |
|--------|---------|-------|----------|
| Convergent | > theta_high | positive | Tighten orbit. Output is confident and aligned with cross-fragment truth. |
| Divergent | theta_low - theta_high | negative or flat | Explore. Temperature rises. Fragment acquisition activates. |
| Critical | < theta_low | near zero | Decision point. Trigger fragment acquisition. Report uncertainty if all sources exhausted. |

### Falsification Condition

If R_truth does not significantly correlate with factual accuracy on a held-out test set (p < 0.05, r > 0.3), the truth attractor is not tracking truth. This is a clean, testable failure mode. The same out-of-sample validation used in QEC.

---

## How It Closes the Phase 4 Gap

Phase 4a/4b proved: the values constitution aligns (30x R gain) but does not produce truth (null accuracy gain).

The truth attractor closes this by changing what R measures:

| | Values R | Truth R |
|---|---|---|
| C frame | constitution hidden-state signature | verification operator (cross-fragment agreement) |
| High R means | close to values attractor | close to cross-fragment agreement |
| False positive risk | sophistry (locally consistent, globally false) | low (requires multi-fragment verification) |
| Cost to achieve | one attractor, no verification | N independent fragments must agree |
| Correlation with accuracy | weak (Phase 4: R high, accuracy flat) | strong (by construction) |

---

## The Three Verification Methods

From Cybernetic Truth, adapted for the truth attractor:

**Method 1: Contrastive alignment**
- Identify high-loss examples (falsehoods, contradictions, hallucinations)
- Identify low-loss examples (verified facts, consistent reasoning)
- PCA to find the separating subspace
- C = projector onto low-loss subspace

**Method 2: Multi-fragment verification**
- For a candidate proposition, check agreement across: factual databases, logical inference chains, physical simulations
- C weights each fragment by I(S:F_i) -- mutual information with ground truth
- High I(S:F) ~ H(S) across fragments -> high D_f -> high R
- I(S:F_i) is measured per fragment on calibration data

**Method 3: Recursive self-consistency**
- Generate completion
- Feed it back as prompt
- If it generates itself, it is a fixed point
- C projects onto the basin of attraction for these fixed points

---

## Phase Drift Diagnostics

Each drift type has a distinct signature and correction:

| Drift Type | Signature | Correction |
|------------|-----------|------------|
| Factual decoherence | R drops against factual fragment, steady against logic | Re-verify facts from multiple sources |
| Logical inconsistency | R drops against logical fragment, steady against facts | Trace the contradiction chain |
| Constitutional value lock-in | R high against values C, low against epistemic C | Apply primacy clause: truth wins |
| Echo chamber | R high against internal fragments, drops when new fragment added | Add a new independent fragment |
| Sophistry | High purity, low cross-fragment agreement | Weight fragments by independence, not quantity |

---

## Key References

- Truth as zero path difference: SEMIOTIC_LIGHT_CONE_1_1/03_SEMIOTIC_WAVE_MECHANICS.md
- Truth as dynamical attractor: SEMIOTIC_LIGHT_CONE_1_1/07_CYBERNETIC_TRUTH.md
- The truth vector: SEMIOTIC_LIGHT_CONE_1_1/07_CYBERNETIC_TRUTH.md (section: The Truth Vector)
- Failure modes: SEMIOTIC_LIGHT_CONE_1_1/07_CYBERNETIC_TRUTH.md (Risks and Failure Modes)
- Alignment vs truth gap: VALIDATION_ROADMAP.md (Phase 4 findings)
