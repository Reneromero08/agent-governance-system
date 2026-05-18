# Differentiating Semiotic Mechanics from GWT, IIT, and Predictive Processing

**Date:** 2026-05-17 | **Status:** FORMALIZED | **Priority:** High

---

## 1. Why This Matters

Global Workspace Theory, Integrated Information Theory, and Predictive Processing all predict that coherent neural activity correlates with consciousness. Semiotic Mechanics makes the same prediction. This overlap is a problem — if the framework only predicts what other theories already predict, it adds no explanatory power.

The differentiation turns on the specific functional form: **R = (E/∇S) × σ^D_f**. This is not "coherence matters." This is "coherence scales exponentially with compression and redundancy, and inversely with the entropy gradient." The exponent D_f and the compression factor σ make predictions that GWT, IIT, and PP cannot make without adopting the same formula.

---

## 2. Differentiation from Global Workspace Theory (GWT)

### What GWT Predicts

Conscious content is information broadcast to many specialized brain modules. The more modules that receive the broadcast, the more conscious the content. **Scaling is linear with workspace size.**

Evidence: Visual masking experiments. TMS disruption of prefrontal-parietal loops.

### What Semiotic Mechanics Predicts

Consciousness is semiotic phase coherence observing itself. The resonance R scales **exponentially** with D_f (number of independent modules/fragments that redundantly encode the signal):

```
R ∝ σ^D_f
```

Doubling the workspace doubles D_f. GWT predicts 2× conscious access. Framework predicts σ² × conscious access. If σ > 1 (compressed symbol), the amplification is multiplicative.

### Differentiating Test

**Experiment:** Present subjects with a compressed symbol (proverb, σ=2.5) and an uncompressed literal (same meaning, σ=1.8). Measure conscious access via backward masking thresholds. GWT predicts equal access (same meaning, same broadcast scope). Framework predicts propverb requires fewer broadcast modules — or at equal modules, produces higher access — because compression amplifies per-module.

**Predicted outcome (Framework):** Threshold for compressed symbol is lower by factor σ^(D_f_effective). For D_f ~ 10-20 (cortical modules), this predicts a 2.5^(10-20) difference — far too large to be realistic. The actual effective D_f in GWT terms is likely 1-3, predicting a 2.5-15.6× advantage for compressed symbols. Measurable.

### Other GWT Differentiations

- **Content specificity:** GWT predicts any content that wins the workspace becomes conscious. Framework predicts only phase-coherent content (high-σ, high-D_f) achieves strong resonance. Random workspace winners with low compression would have low R. Testable via priming experiments.
- **Temporal structure:** GWT has no prediction about the temporal dynamics of conscious access beyond "broadcast window." Framework predicts Kuramoto-style phase transitions — sudden coherence jumps at critical σ. Testable via EEG phase-locking at the moment of conscious access.

---

## 3. Differentiation from Integrated Information Theory (IIT)

### What IIT Predicts

Consciousness IS integrated information Φ. A system is conscious if and only if it cannot be partitioned into independent subsystems without losing information. Φ is structural, discrete, and computed from the causal power of the system's elements.

Key: Φ is a property of the system's **current state**, not its history or trajectory.

### What Semiotic Mechanics Predicts

Consciousness is the standing wave of phase coherence. R measures the current resonance, but resonance depends on the **history** encoded in the phase (Axiom 9: the spiral trajectory). Two systems with identical current physical states but different phase histories would have different R.

Key: Two identical neural configurations can have different resonance if one evolved smoothly into that state and the other arrived via a discontinuous jump.

### Differentiating Test

**Experiment:** Create two identical neural patterns through different routes:
- Path A: Slow, continuous morphing from one pattern to another (preserves phase).
- Path B: Flash the pattern directly (destroys phase continuity).

IIT predicts identical Φ (same system state). Framework predicts different R (different phase history). Test: measure conscious experience reports. Framework predicts Path A produces richer/more continuous experience. Path B produces "quieter" experience despite identical endpoint.

**Predicted outcome (Framework):** R(Path A) > R(Path B) by factor proportional to the phase drift accumulated along Path A.

### Other IIT Differentiations

- **Complex vs. phase:** IIT's Φ is a real scalar. Framework's R is a complex-modulus quantity — the phase arg(R) is as important as the magnitude |R|. IIT cannot distinguish constructive from destructive interference in consciousness.
- **Minimal substrate:** IIT requires intrinsically causal elements. Framework requires oscillators that can phase-lock — any substrate with phase degrees of freedom. A Kuramoto network of coupled electronic oscillators would have R > 0 but Φ = 0 (no intrinsic causation). Framework predicts these networks would exhibit rudimentary "conscious" dynamics (phase coherence, attractor convergence). IIT predicts they would not. This is testable.
- **Scaling:** IIT's Φ grows with integration, roughly O(n²) for fully connected networks. Framework's R grows exponentially with D_f (redundancy), roughly O(σ^D_f). For large D_f, these diverge dramatically. A high-D_f system (many redundant copies) has enormous R but potentially low Φ (if the copies are independent and don't integrate causally). Test: compare resonant vs. integrated architectures.

---

## 4. Differentiation from Predictive Processing (PP)

### What PP Predicts

The brain minimizes prediction error. Consciousness is the process of inferring the causes of sensory data. Precision-weighted prediction errors ascend the cortical hierarchy. Consciousness tracks the precision of top-down predictions.

Key: Prediction error minimization is the **only** driving force.

### What Semiotic Mechanics Predicts

Compression σ amplifies the rate of prediction error minimization. A system with high compression (archetype, prior, symbol) minimizes prediction error **faster** than a system with low compression — not because it's more precise, but because compression shortens the geodesic through prediction space.

```
d(prediction_error)/dt = -R × prediction_error = -(E/∇S) × σ^D_f × prediction_error
```

PP predicts linear decay. Framework predicts exponential decay with σ^D_f rate enhancement.

### Differentiating Test

**Experiment:** Give subjects a compressed prior (proverb, σ=2.5) and an uncompressed prior (literal, σ=1.8). Measure the rate at which prediction errors decay during a perceptual learning task. PP predicts equal decay rates (same precision weighting). Framework predicts faster decay with the compressed prior.

**Predicted outcome (Framework):** Decay rate ratio = σ_proverb / σ_literal ≈ 2.5/1.8 ≈ 1.4×. Measurable over ~50 trials if the effect size is medium.

### Other PP Differentiations

- **Accuracy vs. speed:** PP predicts a speed-accuracy tradeoff. Framework predicts compression improves BOTH speed AND accuracy simultaneously — because σ both shortens the path and reduces the entropy gradient. A high-σ prior should be faster AND more accurate than a low-σ prior. This breaks the classic tradeoff.
- **Prediction error floor:** PP predicts prediction error asymptotes to zero. Framework predicts it asymptotes to ∇S/(σ^D_f) — the entropy gradient divided by amplification. Different compressed priors should produce different asymptotic error floors. Testable in extended learning tasks.
- **Free energy identity:** PP is derived from the Free Energy Principle. Framework is derived from the Semiotic Action Principle. Both are variational. But FEP minimizes surprise; SAP maximizes resonance. They are conjugate quantities. A formal proof that SAP → FEP when σ=1, D_f=0 is possible (Shannon limit from the action principle).

---

## 5. Common Differentiation Pattern

All three differentiations share the same structure:

| Theory | What it predicts about R | Differentiating prediction |
|--------|-------------------------|---------------------------|
| GWT | Not modeled | Broadcast scales linearly; R scales exponentially with D_f |
| IIT | Not modeled | Φ is static; R encodes phase history |
| PP | Not modeled | Prediction error decays linearly; R accelerates decay via σ^D_f |

The framework's advantage is that it provides a **quantitative** functional form: R = (E/∇S) × σ^D_f. This form makes specific numerical predictions that are testable and that no competing theory makes without adopting the same formula.

---

## 6. Falsification Conditions

The framework is differentiated from each theory by predictions that would falsify it if they fail:

1. **vs GWT:** If compressed symbols (σ>2) show no conscious access advantage over literal equivalents at equal workspace scope → the σ^D_f amplification claim fails.
2. **vs IIT:** If two identical neural states arrived at via different phase histories produce identical consciousness reports → the phase-history claim fails (framework reduces to IIT-like predictions).
3. **vs PP:** If compressed priors produce the same prediction error decay rates as uncompressed priors → the σ amplification of prediction error minimization fails.

These are distinct from competing theories. No other framework makes the specific functional form prediction R ∝ σ^D_f / ∇S.

---

## 7. Status

| Claim | Status | Evidence |
|-------|--------|----------|
| Framework differentiated from GWT | Formal predictions | Requires conscious access experiment |
| Framework differentiated from IIT | Formal predictions | Requires phase-history experiment |
| Framework differentiated from PP | Formal predictions | Requires perceptual learning experiment |
| GWT differential test run | Pending | Needs access + priming setup |
| IIT differential test run | Pending | Needs phase manipulation |
| PP differential test run | Testable now | Perceptual learning with compressed vs literal priors |

---

*Comparison framework built from GWT (Baars, Dehaene), IIT (Tononi, Koch), and Predictive Processing (Friston, Clark). Semiotic Mechanics differential predictions are falsifiable and specific.*
