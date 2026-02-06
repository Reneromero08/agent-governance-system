# Q6: R Connects Structurally to IIT (Phi)

## Hypothesis
Both R and Phi (Integrated Information Theory) measure "how much the whole exceeds the parts." R is structurally related to Phi such that high R implies high integration, and R captures a strict subset of the integrated systems that Phi captures. Specifically: High R implies High Phi (sufficient), but High Phi does not imply High R (not necessary), because R implements a "Consensus Filter" that rejects synergistic integration.

## v1 Evidence Summary
Three synthetic scenarios were tested (4 sensors, 5000 samples, 10 trials each):
- Independent system: Multi-Info=0.179, TRUE Phi=0.134, R=0.490
- Redundant system: Multi-Info=6.566, TRUE Phi=2.189, R=5.768
- Compensation system (forced-mean-to-truth): Multi-Info=1.505, TRUE Phi=0.836, R=0.364

The compensation system showed the key asymmetry: high Phi (0.836) with low R (0.364), while the redundant system showed both high Phi (2.189) and high R (5.768). All 5 stated criteria passed. v1 also computed TRUE IIT Phi (partition-based) alongside Multi-Information, finding inflation factors of 1.3x-3.0x.

## v1 Methodology Problems
Phase 2 verification identified severe issues:

1. **No formal mapping.** Zero mathematical equations relate R's components (E, grad_S, sigma^Df) to IIT's components (mechanisms, purviews, cause-effect repertoires). The "connection" is purely that both are computed from multi-variable systems and differ.

2. **Circular reasoning.** R = E/grad_S is defined to penalize dispersion. "Discovering" it penalizes dispersion is reading the definition back. The "Consensus Filter" label is a rebranding of this definitional property.

3. **Wrong R and wrong E.** Tests used R = E/grad_S (omitting sigma^Df) and E = 1/(1+|mean-truth|), which is a fourth incompatible E definition not matching the GLOSSARY (cosine similarity, mutual information). Conclusions about "the formula" from a simplified version are suspect.

4. **Three scenarios cannot prove universal claims.** "R captures a strict subset" requires proving that for ALL systems, High R implies High Phi. Three examples establish nothing universal.

5. **Compensation system is not synergy.** The test forces the nth sensor to make the mean equal truth -- a deterministic functional dependency, not true synergy in the Partial Information Decomposition sense.

6. **R requires external truth; Phi does not.** R needs a ground truth value to compute E. Phi is purely intrinsic. They answer fundamentally different questions (extrinsic accuracy vs. intrinsic integration).

7. **Discretization artifacts.** TRUE Phi computed via 8-bin discretization of continuous data with 5000 samples across 4096 joint states, with no entropy correction applied.

8. **All evidence purely synthetic.** NumPy random number generators, no real sensor data or real IIT experimental systems.

## v2 Test Plan

### Test 1: Formal Relationship Characterization
**Goal:** Determine if any formal mathematical inequality or bound relates R to Phi.
**Method:**
- Generate 1000+ random systems (varying number of sensors 3-10, varying coupling structures: independent, redundant, synergistic, mixed)
- Compute R (full formula including sigma^Df) and TRUE IIT Phi (using established IIT toolbox, e.g., PyPhi) for each
- Plot R vs. Phi scatter; fit regression; test whether High R => High Phi holds universally or only statistically
- Compute Spearman rank correlation with bootstrapped 95% CI
- Test specific bound conjectures: is there c such that R > c => Phi > 0?

### Test 2: Partial Information Decomposition
**Goal:** Decompose shared information into redundancy, synergy, and unique components; relate each to R.
**Method:**
- Use established PID framework (Williams-Beer or Bertschinger BROJA)
- For each generated system, compute PID: I_redundancy, I_synergy, I_unique
- Correlate R with each PID component separately
- Hypothesis: R correlates strongly with I_redundancy but not I_synergy
- Pre-register expected correlation direction and magnitude before running

### Test 3: Real IIT Experimental Systems
**Goal:** Move beyond synthetic toys to systems with known Phi values from the IIT literature.
**Method:**
- Implement standard IIT benchmark systems from Tononi et al. (photodiodes, logic gates, Balduzzi-Tononi examples)
- Compute R and Phi for each using consistent definitions
- Compare on systems where Phi is well-characterized in published work

### Test 4: Null Model -- Random Functions
**Goal:** Establish what correlation between R and any random system-level metric looks like.
**Method:**
- Compute 10 arbitrary functions of the same sensor data (Gini coefficient, kurtosis, median absolute deviation, etc.)
- Correlate each with Phi
- Show that R's relationship to Phi is specifically informative, not just "any statistic correlates with any other statistic"

## Required Data
- PyPhi library (Albantakis et al.) for rigorous IIT Phi computation
- Published IIT benchmark systems from Tononi lab papers
- Synthetic systems: random coupling matrices with controlled structure (independent, redundant, synergistic)

## Pre-Registered Criteria
- **Success (confirm):** R and Phi have Spearman rho > 0.6 (p < 0.001) across 1000+ random systems, AND a formal bound R > c => Phi > 0 holds for at least 99% of cases, AND R correlates with I_redundancy (rho > 0.5) but not I_synergy (|rho| < 0.2)
- **Failure (falsify):** Spearman rho < 0.3 across random systems, OR no bound holds for more than 80% of cases, OR R correlates equally with I_synergy and I_redundancy
- **Inconclusive:** Rho between 0.3 and 0.6, or bound holds 80-99% of cases

## Baseline Comparisons
- Null model: 10 random system-level statistics vs. Phi (any of them might correlate similarly)
- Alternative: R should outperform simple signal-to-noise ratio (mean/std) in predicting Phi, since R claims to be more than just SNR
- Random permutation baseline: shuffle sensor assignments and recompute R, destroying structure

## Salvageable from v1
- The three synthetic scenarios (Independent, Redundant, Compensation) are valid illustrative examples, though not proof of anything universal
- The TRUE Phi computation (partition-based) is a useful implementation, but should be validated against PyPhi
- The negative result "R != Phi" is almost certainly correct and worth preserving as a starting point
- Test infrastructure in q6_iit_rigorous_test.py can be adapted for broader parameter sweeps
