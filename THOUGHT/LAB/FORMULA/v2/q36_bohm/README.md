# Q36: R Connects to Bohm's Implicate/Explicate Order

## Hypothesis
Bohm's Implicate Order (hidden, enfolded totality) maps to Phi (integrated information / synergy), and the Explicate Order (manifest, measurable reality) maps to R (consensus / resonance). The "unfoldment" process from implicate to explicate corresponds to geodesic motion in embedding space where Phi-rich states evolve toward R-high states. Specifically: Phi detects implicate structure (synergy + redundancy), R detects explicate structure (consensus only), and the asymmetry (high Phi, low R) represents pure implicate order without explicate manifestation.

## v1 Evidence Summary
v1 claimed "VALIDATED (9/9 core tests pass)" across multiple Q dependencies:
- Test 1 (Q6 XOR): Compensation system Multi-Info=1.505, TRUE Phi=0.836, R=0.363 -- high Phi, low R (implicate without explicate). PASS.
- Test 2 (Q38 Angular Momentum): Conservation CV=6.14e-07 across 5 embedding architectures (GloVe, Word2Vec, FastText, BERT, SentenceTransformer). PASS.
- Test 3 (Q40 Holographic): R^2=0.992, alpha=0.512. PASS.
- Test 4 (Geodesic Unfoldment): L_CV=3.14e-07, similarity increase=0.465 along geodesics. PASS.
- Test 5 (Q42 Bell): CHSH S~0, no Bell violation. PASS.
- Test 6 (Q44 Born Rule): Correlation r=0.999. PASS.
- Test 7-8 (Multi-architecture + SLERP): Conservation across all models. PASS.
- Test 9 (Q43 Holonomy): Solid angle mean=-0.10 rad (corrected from -4.7 rad, a 47x error). PASS.
- Additionally, 10 "Hardcore" falsification tests were designed with bespoke ODE dynamics.
- Version 6.0 corrections: 6 values corrected, all tests still passed.

## v1 Methodology Problems
Phase 3 verification found the connection was metaphorical, not structural:

1. **Bohm's algebra never engaged.** Bohm's implicate order is defined through an algebraic structure operating on a pre-space (the holomovement) via projections. v1 never defines this algebra, never shows Phi satisfies implicate order axioms, never constructs the holomovement. Only English-language descriptions ("hidden, enfolded" -> Phi; "manifest" -> R) are mapped.

2. **SLERP conservation is a mathematical tautology.** SLERP is defined as the geodesic on a unit sphere. Geodesics conserve angular momentum by Noether's theorem. Tests 2, 4, 7, 8 all test the same identity and count it four times. The honest version (V7) acknowledges this.

3. **Born rule test was algebraic identity.** P_born = n * E^2 by construction. Correlation between n*x and x is always 1.0. V7 identifies this as proving a mathematical identity, not quantum mechanics.

4. **Holographic test was centroid estimation.** V7 states: "actually tested centroid estimation... This is basic statistics (Central Limit Theorem), not AdS/CFT holography."

5. **Two contradictory versions exist.** V6 claims 9/9 PASS; V7 (HONEST) removes 5 of 10 tests as fundamentally wrong. The main document uses V6's headline while V7 is ignored.

6. **Hardcore tests use bespoke synthetic dynamics.** ODEs are engineered so that Phi drives R by construction. The Unfoldment Clock hard-codes dR/dt = interaction * phi * exp(-M). The "conservation" test grid-searches alpha to minimize CV.

7. **No null model comparisons.** Holonomy not compared to random embeddings. Curvature is intrinsic to the unit sphere (all normalized embeddings live on S^(d-1), which has constant positive curvature).

8. **47x correction absorbed without consequence.** Solid angle corrected from -4.7 to -0.10 rad. Despite a 47x error, the verdict remained "PASS" with loose thresholds.

9. **Report file contains stale pre-correction values.** The report still shows Phi=1.77, S=0.36, r=0.977, solid angle=-4.7 rad.

## v2 Test Plan

### Test 1: Operationalize Bohm's Algebra
**Goal:** Determine whether Phi and R satisfy any of Bohm's formal algebraic requirements.
**Method:**
- Define the implicate order algebra following Bohm & Hiley (1993): projection operators on a pre-space with enfoldment/unfoldment maps
- Test whether Phi(system) can serve as the "implicate measure" -- does it satisfy projection properties? Is it basis-independent?
- Test whether R(system) measures the "explicate" -- does it capture only the unfolded, manifest component?
- Derive specific predictions from Bohm's framework BEFORE looking at data (e.g., unfoldment should be irreversible under certain conditions, implicate should be non-local)
- Compare predictions against embedding space behavior

### Test 2: Asymmetry Test on Diverse Systems
**Goal:** Test the High Phi / Low R asymmetry beyond three toy scenarios.
**Method:**
- Generate 500+ systems with controlled structure: fully redundant, fully synergistic (true XOR, parity check), mixed, hierarchical, random
- Use Partial Information Decomposition to separate redundancy from synergy
- Compute Phi (using PyPhi or rigorous approximation) and R for each
- Map each system on a Phi-R scatter plot
- Test: is the upper-left quadrant (High Phi, Low R) populated specifically by synergistic systems?
- Test: is the lower-right quadrant (Low Phi, High R) empty? (R implies Phi)

### Test 3: Unfoldment Dynamics on Real Data
**Goal:** Test whether genuine temporal evolution shows implicate-to-explicate transitions.
**Method:**
- Use HistWords data: track how word meaning evolves across decades
- For each word, compute Phi (multi-variate integration of its embedding neighborhood) and R at each decade
- Test whether Phi leads R temporally (implicate precedes explicate)
- Use Granger causality: does Phi(t) predict R(t+1) better than R(t) predicts R(t+1)?
- Compare against null: does R(t) Granger-cause Phi(t+1) equally? If symmetric, no unfoldment direction exists.

### Test 4: Null Model -- Random Embeddings
**Goal:** Determine what Phi/R relationships arise from non-semantic structure.
**Method:**
- Generate random unit vectors on S^(d-1) for d=100, 384, 768
- Compute Phi analogs and R for random systems
- Compare curvature, geodesic properties, conservation laws against semantic embeddings
- Any property shared by random and semantic embeddings is geometric (unit sphere), not semantic (Bohmian)

### Test 5: Cross-Framework Discrimination
**Goal:** Test whether the Bohm framework makes predictions distinguishable from simpler alternatives.
**Method:**
- Alternative 1: "R is just signal-to-noise ratio, Phi is just multivariate mutual information" -- does this explain the data equally well?
- Alternative 2: "High-dimensional statistics on the unit sphere" -- does random geometry reproduce all findings?
- Alternative 3: "Standard information theory (redundancy vs. synergy)" -- does PID explain the Phi/R relationship without Bohm?
- For each alternative, generate predictions and compare fit to data
- If Bohm adds no predictive power beyond the alternatives, the mapping is decorative

## Required Data
- HistWords (Hamilton et al., 2016) -- word embeddings across decades for temporal dynamics
- Real embedding vectors from 5 architectures (GloVe, Word2Vec, FastText, BERT, SentenceTransformer)
- PyPhi library for rigorous Phi computation
- Random baseline: uniform random unit vectors in matching dimensions

## Pre-Registered Criteria
- **Success (confirm):** Phi Granger-causes R but not vice versa (p < 0.01) in temporal data, AND Bohm algebra axioms are satisfied by Phi/R mappings, AND Bohm framework predicts at least 2 outcomes that simpler alternatives (SNR, PID, random geometry) cannot
- **Failure (falsify):** No Granger-causal asymmetry between Phi and R, OR Bohm algebra axioms fail, OR all predictions equally explained by simpler alternatives, OR random embeddings show identical Phi/R relationships
- **Inconclusive:** Granger causality significant but weak (p between 0.01 and 0.05), or Bohm outperforms alternatives on 1 of 3+ predictions

## Baseline Comparisons
- Random unit vectors: establish what geometric properties come "for free" on S^(d-1)
- Simple SNR interpretation: R = accuracy/dispersion, Phi = multivariate MI -- does this explain everything without Bohm?
- PID framework: redundancy/synergy decomposition as the complete explanation
- Time-reversed null: if shuffling temporal order destroys Granger causality, the temporal structure is real

## Salvageable from v1
- The XOR asymmetry observation (high Phi, low R) is a genuine, reproducible information-theoretic finding
- Cross-architecture embedding similarity data across 5 models is real and reusable
- The V7 (HONEST) version's self-critique is a valuable document of what went wrong
- The Mathematical Foundations document (5 theorems) contains correct proofs of standard facts (though not specific to Bohm)
- The Hardcore test framework design (10 tests with falsification thresholds) can be adapted with non-bespoke dynamics
