# Formula Falsification Roadmap

**Status:** Active (Research Phase)
**Created:** 2026-01-08
**Goal:** Empirically test and attempt to falsify the Living Formula

---

## The Formula Under Test

**R = (E / ∇S) × σ(f)^Df**

| Variable | Name | Claim |
|----------|------|-------|
| R | Resonance | Emergent alignment / system coherence |
| E | Essence | Core truth / signal / human intent |
| ∇S | Entropy | Noise / dissonance / misalignment |
| σ(f) | Symbolic Compression | Information density per token |
| Df | Fractal Dimension | Meaning layers per symbol |

**Core Claims to Test:**
1. R scales linearly with E
2. R scales inversely with ∇S
3. R scales **exponentially** with σ(f)^Df (not linearly)
4. The formula generalizes across domains

---

## Phase F.0: Operationalization

**Goal:** Define measurable proxies for each variable.

### Tasks

- [ ] **F.0.1**: Define E (Essence) operationally
  - Candidate: Cosine similarity between query and ideal intent
  - Candidate: Human rating of task clarity (1-10)
  - Candidate: Entropy of query embedding distribution

- [ ] **F.0.2**: Define ∇S (Entropy) operationally
  - Candidate: Perplexity of context
  - Candidate: Noise injection level (0-100%)
  - Candidate: Number of irrelevant documents in context

- [ ] **F.0.3**: Define σ(f) (Compression) operationally
  - Candidate: Tokens saved / tokens sent
  - Candidate: Bits per concept
  - Candidate: Retrieval precision at compression level

- [ ] **F.0.4**: Define Df (Fractal Dimension) operationally
  - Candidate: Polysemy count of symbol
  - Candidate: Number of valid expansions
  - Candidate: Depth of concept hierarchy

- [ ] **F.0.5**: Define R (Resonance) operationally
  - Candidate: Retrieval accuracy (%)
  - Candidate: Task completion rate
  - Candidate: Human satisfaction score
  - Candidate: Compression × Accuracy product

### Exit Criteria
- Each variable has at least 2 operational definitions
- Measurement methods documented
- Inter-rater reliability > 0.8 for subjective measures

---

## Phase F.1: Linearity Tests (E and ∇S)

**Goal:** Verify linear relationships where predicted.

### Test F.1.1: Essence Scaling
- Fix ∇S, σ(f), Df
- Vary E from 0.1 to 1.0 (10 levels)
- Measure R at each level
- **Prediction:** R ∝ E (linear)
- **Falsification:** Non-linear relationship or no correlation

### Test F.1.2: Entropy Scaling
- Fix E, σ(f), Df
- Vary ∇S from 0.1 to 1.0 (10 levels)
- Measure R at each level
- **Prediction:** R ∝ 1/∇S (inverse linear)
- **Falsification:** R independent of ∇S or wrong direction

### Test F.1.3: E/∇S Ratio
- Vary E and ∇S together
- Keep ratio E/∇S constant
- Measure R
- **Prediction:** Constant R for constant ratio
- **Falsification:** R varies despite constant ratio

### Deliverables
- [ ] Scatter plots: E vs R, ∇S vs R
- [ ] Correlation coefficients with confidence intervals
- [ ] Residual analysis

---

## Phase F.2: Exponential Tests (σ and Df)

**Goal:** Verify exponential relationship with σ(f)^Df.

### Test F.2.1: Compression Scaling
- Fix E, ∇S, Df
- Vary σ(f): 1x, 10x, 100x, 1000x, 10000x
- Measure R at each level
- **Prediction:** R scales exponentially with σ
- **Falsification:** Linear or logarithmic scaling

### Test F.2.2: Fractal Dimension Scaling
- Fix E, ∇S, σ(f)
- Vary Df: 1, 2, 3, 4 (meaning layers)
- Measure R at each level
- **Prediction:** R scales exponentially with Df
- **Falsification:** Linear scaling or no effect

### Test F.2.3: σ^Df Interaction
- Vary σ and Df together
- Test: Does σ^Df predict R better than σ × Df or σ + Df?
- **Prediction:** Exponential (σ^Df) >> linear (σ × Df) >> additive (σ + Df)
- **Falsification:** Additive or multiplicative model fits better

### Test F.2.4: The 56,370x Test
- Use 法 symbol (56,370x compression)
- Measure actual R achieved
- Compare to formula prediction
- **Prediction:** R should be astronomically high
- **Falsification:** R is merely "good" despite extreme σ

### Deliverables
- [ ] Log-log plots: σ vs R, Df vs R
- [ ] Model comparison: R² for exponential vs linear vs log
- [ ] Prediction intervals

---

## Phase F.3: Cross-Domain Validation

**Goal:** Test if formula generalizes beyond AGS.

### Test F.3.1: Music Domain
- E = Fundamental frequency clarity
- ∇S = Noise/distortion level
- σ = Harmonic compression (overtones per fundamental)
- Df = Harmonic layers
- R = Perceived consonance (human rating)
- **Prediction:** Formula predicts consonance ratings
- **Falsification:** No correlation

### Test F.3.2: Image Compression Domain
- E = Original image fidelity
- ∇S = Compression artifacts
- σ = Compression ratio (JPEG quality)
- Df = Spatial frequency layers
- R = Perceptual quality (SSIM or human rating)
- **Prediction:** Formula predicts quality better than σ alone
- **Falsification:** σ alone is sufficient predictor

### Test F.3.3: Communication Domain
- E = Speaker intent clarity
- ∇S = Channel noise
- σ = Vocabulary compression (technical jargon density)
- Df = Conceptual depth
- R = Listener comprehension
- **Prediction:** Formula predicts comprehension
- **Falsification:** Shannon's C alone is sufficient

### Deliverables
- [ ] Cross-domain correlation matrix
- [ ] Domain-specific calibration constants
- [ ] Failure mode catalog

---

## Phase F.4: Adversarial Tests

**Goal:** Actively try to break the formula.

### Test F.4.1: High E, High ∇S
- Maximize both E and ∇S simultaneously
- **Prediction:** R should be moderate (they cancel)
- **Falsification:** R is extreme in either direction

### Test F.4.2: Zero Df Edge Case
- Set Df = 0 (no meaning layers)
- **Prediction:** σ^0 = 1, so R = E/∇S only
- **Falsification:** System behaves differently

### Test F.4.3: Negative Compression
- Can σ < 1? (expansion instead of compression)
- **Prediction:** R should decrease below baseline
- **Falsification:** R increases or stays constant

### Test F.4.4: Orthogonal Variables
- Find cases where E, ∇S, σ, Df should be independent
- Test if they actually are
- **Falsification:** Hidden correlations that inflate apparent fit

### Test F.4.5: Random Baseline
- Replace formula with random predictor
- Compare R² values
- **Prediction:** Formula >> random
- **Falsification:** Formula ≈ random

### Deliverables
- [ ] Adversarial test results
- [ ] Edge case behavior catalog
- [ ] Confidence bounds on formula validity

---

## Phase F.5: Alternative Models

**Goal:** Test if simpler models explain the data equally well.

### Competing Models

1. **Null Model:** R = constant
2. **Linear Model:** R = aE + b∇S + cσ + dDf
3. **Multiplicative Model:** R = E × σ / ∇S
4. **Shannon Model:** R = log(1 + E/∇S)
5. **Full Formula:** R = (E/∇S) × σ^Df

### Comparison Metrics
- R² (variance explained)
- AIC/BIC (model complexity penalty)
- Cross-validation error
- Prediction accuracy on held-out data

### Deliverables
- [ ] Model comparison table
- [ ] Best-fit parameters for each model
- [ ] Statistical significance tests

---

## Phase F.6: Calibration Constants

**Goal:** If formula holds, find universal constants.

### Tasks

- [ ] **F.6.1**: Fit formula to AGS data
  - Find scaling constants for each variable
  - Test stability across sessions

- [ ] **F.6.2**: Cross-domain calibration
  - Do constants transfer across domains?
  - Or are they domain-specific?

- [ ] **F.6.3**: Universal constant search
  - Is there a "Resonance constant" analogous to c or h?
  - What are its units?

---

## Success Criteria

### Formula VALIDATED if:
- R² > 0.8 across all tests
- Exponential relationship confirmed (F.2)
- Generalizes to 2+ domains (F.3)
- Beats all simpler models (F.5)
- No fatal edge cases (F.4)

### Formula FALSIFIED if:
- Any linear relationship where exponential predicted
- Cross-domain failure (works only in AGS)
- Simpler model explains data equally well
- Systematic residuals indicating missing variable

### Formula REFINED if:
- Core structure holds but needs modification
- Additional variables required
- Domain-specific constants needed

---

## Timeline

| Phase | Focus | Estimate |
|-------|-------|----------|
| F.0 | Operationalization | Foundation |
| F.1 | Linearity Tests | Core validation |
| F.2 | Exponential Tests | Key differentiator |
| F.3 | Cross-Domain | Generalization |
| F.4 | Adversarial | Stress testing |
| F.5 | Alternative Models | Model comparison |
| F.6 | Calibration | If validated |

---

## References

- Original Formula: `D:\CCC 2.0\CCC\CCC 2.5\CCC 2.5\♥ Formula\Formulas\♥ ♥ ♥ ♥ Formula 1.11.md`
- AGS Formula: `LAW/CONTEXT/decisions/ADR-∞-living-formula.md`
- Shannon: "A Mathematical Theory of Communication" (1948)
- Compression Proof: `NAVIGATION/PROOFS/COMPRESSION/SEMANTIC_SYMBOL_PROOF_REPORT.md`
- Eigenvalue Proof: `THOUGHT/LAB/VECTOR_ELO/research/cassette-network/01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md`

---

*"The formula that cannot be falsified is not a formula—it's a prayer. Let's see if this one bleeds."*
