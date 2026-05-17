# The Truth Attractor

## What This Is

The truth attractor is a **verification operator** -- an alignment frame C built from cross-fragment agreement instead of value similarity. It replaces the values constitution as the primary attractor for the Cybernetic Truth control loop.

Where the values constitution says "orbit this ethical center," the truth attractor says "this is how you know the center is real."

## The Core Mechanism

### The Epistemic C Frame

C_epistemic = (1/N) * sum_{i=1}^{N} P_i

Where:
- P_i is the projection onto the verification subspace of fragment i
- N is the number of independent verification fragments
- Each fragment is an independent channel: factual database, logical inference chain, physical simulation, cross-model agreement, recursive self-consistency

When all fragments agree (zero path difference), C_epistemic is a rank-1 projector onto the agreed truth. R = Tr(rho * C) is maximized.

When fragments disagree (path difference > 0), C_epistemic is mixed. R drops. The system enters divergent regime: explore, seek resolution, gather more fragments.

### Truth-Tracking R

R_truth = Tr(rho * C_epistemic)

This measures proximity to the truth attractor -- zero path difference across independent observers. It is NOT the same as alignment R (which measures proximity to a values attractor).

### The Control Law

T = 1 / (R_truth + epsilon)

- High R (all fragments agree) -> low T -> deterministic, confident output
- Low R (fragments disagree) -> high T -> exploratory, seeking resolution
- R near zero (total disagreement, decoherence) -> halt. Do not output when you cannot verify.

### Three Regimes

| Regime | R_truth | dR/dt | Behavior |
|--------|---------|-------|----------|
| Convergent | > 0.7 | positive | Tighten orbit. Output is confident and aligned with cross-fragment truth. |
| Divergent | 0.3 - 0.7 | negative or flat | Explore. Temperature rises. Seek new fragments. Gather more evidence. |
| Critical | < 0.3 | near zero | Decision point. The system is at a saddle. External input or new fragments break symmetry. |

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

## The Three Verification Methods

From Cybernetic Truth, adapted for the truth attractor:

**Method 1: Contrastive alignment**
- Identify high-loss examples (falsehoods, contradictions, hallucinations)
- Identify low-loss examples (verified facts, consistent reasoning)
- PCA to find the separating subspace
- C = projector onto low-loss subspace

**Method 2: Multi-fragment verification**
- For a candidate proposition, check agreement across: factual databases, logical inference chains, physical simulations
- C weights each fragment by independent verification strength
- High mutual information I(S:F) ~ H(S) across fragments -> high Df -> high R

**Method 3: Recursive self-consistency**
- Generate completion
- Feed it back as prompt
- If it generates itself, it is a fixed point
- C projects onto the basin of attraction for these fixed points

## Phase Drift Diagnostics

Each drift type has a distinct signature and correction:

| Drift Type | Signature | Correction |
|------------|-----------|------------|
| Factual decoherence | R drops against factual fragment, steady against logic | Re-verify facts from multiple sources |
| Logical inconsistency | R drops against logical fragment, steady against facts | Trace the contradiction chain |
| Constitutional value lock-in | R high against values C, low against epistemic C | Apply primacy clause: truth wins |
| Echo chamber | R high against internal fragments, drops when new fragment added | Add a new independent fragment |
| Sophistry | High purity, low cross-fragment agreement | Weight fragments by independence, not quantity |

## Key References

- Truth as zero path difference: SEMIOTIC_LIGHT_CONE_1_1/03_SEMIOTIC_WAVE_MECHANICS.md
- Truth as dynamical attractor: SEMIOTIC_LIGHT_CONE_1_1/07_CYBERNETIC_TRUTH.md
- The truth vector: SEMIOTIC_LIGHT_CONE_1_1/07_CYBERNETIC_TRUTH.md (section: The Truth Vector)
- Failure modes: SEMIOTIC_LIGHT_CONE_1_1/07_CYBERNETIC_TRUTH.md (Risks and Failure Modes)
- Alignment vs truth gap: VALIDATION_ROADMAP.md (Phase 4 findings)
