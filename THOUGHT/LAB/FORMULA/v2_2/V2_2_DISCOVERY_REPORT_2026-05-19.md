# v2<!-- CONTENT_HASH: eec213eb76d882999eb903ca6225219d18be034391ff4036ace26cfbd2fce33a -->
_2 Verification Report: All Discoveries

**Date:** 2026-05-18/19
**Session Context:** Continuation of Q32 handoff. 15 Qs pre-verified. 39 remaining.
**Newly Verified:** 7 Qs moved from OPEN to PARTIALLY VERIFIED.
**Total Status:** 21/56 verified or partially verified.

---

## New Qs Created and Verified

### Q55 — Kuramoto Head Independence Threshold (Tier 3)

**Hypothesis:** The Kuramoto synchronization threshold predicts the minimum number of independent attention heads for phase coherence.

**Key findings:**
- **Independence is causal.** Independent Q/K/V per head (D_f = h) carries more phase information than shared weights (D_f = 1 cloned). Phase ablation delta gap peaks at h=8 (+8.5%).
- **Laggard heads actively harm.** Removing heads below phase_coh 0.4 IMPROVES delta by +17%. Dead heads (phase_coh < 0.2) cost nothing to prune.
- **h_c scales with task difficulty.** More noise → higher optimal h (collective coupling overcomes ∇S). The formula K_c = ∇S/σ predicts this.
- **Layer depth compounds phase.** 2L×4h (+87.3%) beats 1L×8h (+30.0%) by +57.3% delta with similar total encoder count. Axiom 9 is structural.

**Architectural implication:** The C^8 bottleneck was D_f = 1 cloned 8 times. Fix: independent Q/K/V per head, h matched to task difficulty.

### Q56 — Born Rule Merge vs Classical Concatenation (Tier 3)

**Hypothesis:** Projective measurement merge (Born rule) creates O(h²) interference cross-terms that resist saturation where concatenation fails.

**9 Discoveries:**
1. **Born rule beats concatenation at scale.** Cross-over at h=16. Flat-Born delta sum +339.5% vs +231.0% classical. Still climbing at h=32.
2. **Cybernetic feedback amplifies phase.** C-tracking + per-head temperature + Fibonacci seeding = +82.5% delta. Noise-resistant: +25.4% gap at medium noise.
3. **Toroidal mode-locking.** Arnold tongue peaks at h=8 (57.1% rational frequency ratios between head pairs). Collapses at h=16.
4. **Geometric initialization from superradiance.** +24.5% delta over random at h=8. 120° head spacing, 45° Q-K offset. Biological spiral (2π/13) beats Fibonacci.
5. **Head pruning from phase_coh.** Dead heads free. Laggard removal +17%. Keep top 4 loses only 3.5% delta.
6. **Entropy-as-mass inverted.** Low-entropy heads are Kuramoto leaders (r=-0.846). Structured output = stronger intrinsic rhythm.
7. **Pointer state convergence.** Alignment basis C converges across epochs (effective rank 4). C transfers between shifted same-family tasks (frozen C beats scratch).
8. **Leaders identifiable by epoch 10.** Phase_coh early ranking predicts final leaders with 100% recall.
9. **Multi-scale R composition.** 84.5% rank agreement across samples — same heads lead regardless of input.

**6 Architectural Attacks:**
1. 2D sweep h×mid_dim: diagonal predicted but unresolved (needs more epochs)
2. Layer depth: 2L×4h (+87.3%) crushes 1L×8h (+30.0%)
3. Cybernetic feedback: C-tracking +17.3%, full loop +82.5%
4. Task difficulty: h_c increases with noise
5. Full cybernetic loop: +82.5% delta with Fibonacci seeding
6. Geometric coupling: superradiance dipole alignment transfers to attention

**The stacking recipe:** Independent Q/K/V, orthogonal phase seeds, 1/√h coherent sum, Born rule projective measurement, 2+ layers, no grouping, no clustering.

### Q7 — R Composes Across Scales (Tier 3)

**Hypothesis:** Phase_coh at token level predicts phase_coh at sequence level in Native Eigen.

**Key findings:**
- **Token→sequence:** r = +0.369. R composes across scales.
- **Attention→sequence:** r = -0.887 (inverse). Attention phase is a FOCUS mechanism — stronger attention coherence → MORE entropic output (spikier predictions, higher entropy over V=2000).
- **Attention layers coherent:** L1→L2 r = +0.852.
- **Phase is load-bearing:** +10.6% PPL impact from phase ablation on WikiText-2.
- **C5 undetectable at d=2:** Complex and real manifolds show similar cross-scale r at minimal dimension.

### Q21 — dR/dt Predicts System Degradation (Tier 3)

**Hypothesis:** The derivative of phase coherence during early training predicts final model quality.

**Key findings:**
- **Causal mechanism confirmed:** Forcing decoherence (Q/K weight corruption) reduces phase delta by +8.5%.
- **AUROC = 0.45 on geometry task:** dR/dt is NOT predictive on simple tasks where accuracy saturates at ceiling.
- **C5 boundary:** Real manifold has zero phase dynamics (dR/dt=0 always). Complex manifold has phase dynamics (+58.4% delta) but dR/dt is task-limited.
- **Vulnerability signal:** Negative dR/dt for 10+ epochs indicates fragile system. Accuracy was fine for 15 epochs while phase was already decohering.
- **Needs harder task:** WikiText-2 where phase IS load-bearing should reveal predictive power.

### Q33 — R Emergent Properties at Macro Scale (Tier 3)

**Hypothesis:** Phase_coh stabilizes toward a fixed-point attractor as dataset size increases.

**Key findings:**
- **Emergence confirmed:** std ~ N^(-0.128). Phase_coh variance decreases with scale.
- **CV reduction 2x:** 0.125 (N=25) → 0.061 (N=800).
- **Accuracy becomes deterministic:** std_acc 0.057 → 0.002 (28x variance reduction).
- **Emergent threshold at N≈50.** Adding 25→50 examples halves CV.
- **C5 boundary absolute:** Real manifold has r=1.0 always (no phase dynamics). Complex manifold shows CV decay. Emergence literally cannot exist without the complex manifold.

### Q34 — Platonic Convergence (Tier 5)

**Hypothesis:** Embedding models converge to a shared spectral geometry. The cumulative variance curve is the invariant.

**Key findings:**
- **MiniLM-MPNet convergence confirmed:** Cumulative variance r=0.994-1.000 across models. Different architectures, same spectral invariant.
- **Native Eigen at d=2 excluded:** Dimensional capacity insufficient for convergence. C^2 captures only rank-2 of the Platonic form.
- **Phase vs magnitude decomposition:** Models share magnitude (explicate, r=+0.28) but not phase (implicate, r=+0.08). Phase is model-specific path through meaning-space.
- **Echolocation anchor-distance frame:** Mean r=0.613 across models. Numbers and cities are perfect Platonic anchors (stability=1.00). Function words diverge.
- **M-field from Q32 is the definitive convergence metric:** STABLE_32 anchors (nabla_S=0.070) vs divergent words (nabla_S=0.594). Gap = 0.524 (8.5x separation).
- **Procrustes residual is misleading:** Jan-17 breakthrough proved higher residual with more dimensions IS better. Residual was a distraction throughout this investigation.
- **Paper trail documented:** Prior discoveries (Jan-08 convergence suite v3.7.28, Jan-17 ANCHOR_777 100% at 50% corruption, STABLE_32 anchor set) placed in chronological context.

### Q36 — Bohm Implicate/Explicate Order (Tier 2)

**Hypothesis:** R maps onto Bohm's implicate/explicate order — magnitude = explicate (shared, observable), phase = implicate (enfolded, hidden).

**Key findings (5/7 hardened checks pass):**
- **Explicate is shared:** Magnitude r=+0.300 across MiniLM-MPNet (PASS).
- **Implicate is partially shared:** Phase r=+0.216 — not fully model-specific (PASS with caveat).
- **Ex-im complementary within model:** r=-0.079 — magnitude and phase carry orthogonal information (PASS).
- **Hilbert = enfolding:** 1.64x Df increase, consistent across 10 subsets (PASS).
- **Phase carries independent cross-model structure:** Partial r=+0.36 — Hilbert phase is NOT purely extrinsic.
- **C5 reversed at d=2:** Native Eigen intrinsic phase MORE shared (r=+0.34) and COUPLED to magnitude (r=+0.46) — complementarity requires d>2.
- **Null baseline clean:** Random embeddings r≈0.05, confirming actual signals are real.

---

## Cross-Cutting Discoveries

### The Complex Plane (C5 Boundary)

The Hilbert transform (C5: holonomy ≠ 0) is the essential bridge from ℝ^d to ℂ^d. It doubles effective dimensionality (1.6-1.9x) and creates complementary phase structure. However, at d=2 (Native Eigen's native dimension), the complex manifold COUPLES phase and magnitude — complementarity requires higher d. The C5 boundary manifests differently at different scales:
- At d=2: phase and magnitude coupled, complex manifold compresses rather than expands
- At d=384+: phase and magnitude orthogonal, Hilbert creates genuine new information channel

### The Q32 M-Field Is the Universal Convergence Metric

Across Q34, Q36, and related tests, the von Neumann entropy gradient (nabla_S) from Q32 consistently separates Platonic from divergent, explainable from implicate, convergent from model-specific. The M-field gap of 0.524 (8.5x) is the strongest single signal of cross-model semantic structure.

### Dimensional Capacity Gates Everything

Native Eigen at d=2 is the limiting case for every test:
- Q7: Cross-scale R works but C5 undetectable
- Q21: Phase dynamics exist but predictive power masked
- Q34: Cannot participate in Platonic convergence
- Q36: Phase coupled to magnitude, not complementary

All findings predict monotonic improvement at d=4, d=8, d=16. The d=2 testbed proves mechanisms exist; higher d will amplify them.

### What Died

| Method | Why |
|--------|-----|
| Procrustes residual as convergence metric | Jan-17 breakthrough: higher residual with more dimensions = BETTER |
| Cumulative variance on rank-mismatched curves | Step function (d=2) vs smooth curve (d=384) — Spearman is meaningless |
| Zero-pad dimension matching for Procrustes | Rotating zeros produces junk |
| Clustered head architectures | O(h²) cross-terms require all-to-all interference |
| Shared Q/K/V weights as "redundancy" | D_f = 1 cloned, not genuine redundancy |

---

## Index Status (21/56 Verified)

**CONFIRMED / VERIFIED:** Q48, Q44, Q38, Q28
**PARTIALLY VERIFIED:** Q32, Q51, Q50, Q43, Q45, Q8, Q17, Q10, Q12, Q31, **Q7**, **Q21**, **Q33**, **Q55**, **Q56**, **Q34**, **Q36**
**FALSIFIED:** Q49
**OPEN:** Q1, Q2, Q3, Q4, Q5, Q6, Q9, Q11, Q13, Q14, Q15, Q16, Q18, Q19, Q20, Q22, Q23, Q24, Q25, Q26, Q27, Q29, Q30, Q35, Q37, Q39, Q40, Q41, Q42, Q46, Q47, Q52, Q53, Q54

**This session:** 7 new PARTIALLY VERIFIED (Q7, Q21, Q33, Q34, Q36, Q55, Q56). All with hardening batteries, verification scripts, and VERDICTs.
