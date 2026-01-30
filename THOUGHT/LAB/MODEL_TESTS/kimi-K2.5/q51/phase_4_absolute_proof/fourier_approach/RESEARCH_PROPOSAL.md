# Q51 Phase 4: Fourier/Spectral Analysis for Absolute Proof

## Research Proposal: Detecting Phase Structure in Semantic Embeddings

**Date:** 2026-01-30  
**Status:** PROPOSED  
**Target:** THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/q51/phase_4_absolute_proof/fourier_approach/  
**Significance:** P < 0.00001 threshold for absolute certainty

---

## 1. Executive Summary

This proposal outlines a rigorous Fourier/spectral methodology to definitively answer Q51: whether real embeddings are shadows of a fundamentally complex-valued semiotic space. Unlike previous geometric tests, this approach uses frequency-domain analysis to detect hidden periodic structures that would be unmistakable signatures of complex phase organization.

**Key Innovation:** Transform embeddings from spatial domain to frequency domain, where complex phase structure manifests as specific spectral signatures impossible to produce by real-valued random processes.

---

## 2. Theoretical Framework

### 2.1 Core Hypothesis

**H1 (Complex Space):** Embeddings are projections: z → Re(z) where z ∈ C^d. This implies underlying frequency structure with specific symmetries.

**H0 (Purely Real):** Embeddings are x ∈ R^d with no complex structure. Spectral properties match random Gaussian processes.

### 2.2 Fourier Signatures of Complex Structure

Complex-valued signals exhibit distinct spectral properties:

| Property | Complex Signal | Real Signal |
|----------|---------------|-------------|
| **Spectrum symmetry** | Asymmetric | Symmetric |
| **Phase coherence** | Global coherence | Local only |
| **Harmonic relations** | Rational multiples | No structure |
| **Cross-frequency coupling** | Phase-amplitude | None |
| **Spectral peaks** | At rational fractions | Uniform/Gaussian |

### 2.3 Mathematical Foundation

**Fourier Transform of Projected Complex Signal:**

Given z(t) = r(t)e^(iθ(t)), the real projection is:
x(t) = Re(z(t)) = r(t)cos(θ(t))

The Fourier transform reveals:
X(ω) = 1/2[R(ω) ⊗ (δ(ω-ω_0) + δ(ω+ω_0))] for constant θ

**Key Insight:** Projections of complex signals create **mirror-symmetric spectra with phase-encoded sidebands**. The sideband structure encodes the original phase information in a recoverable form.

**Autocorrelation Signature:**

For complex-valued processes:
R_xx(τ) = E[x(t)x(t+τ)] = 1/2E[r^2]cos(Δθ)

This creates oscillatory autocorrelation with period related to characteristic phase velocity.

### 2.4 Predicted Spectral Signatures

If embeddings are complex projections, we expect:

1. **Octant Periodicity (8-fold):** Power concentrated at frequencies f, f/2, f/4, f/8
2. **Phase Gradient Coherence:** Cross-spectral density shows phase locking
3. **Fractal Dimension 8e:** Power-law scaling P(f) ~ f^(-α) with α ≈ 1/(8e)
4. **Berry Phase Winding:** Spectral phase exhibits 2π holonomy around degeneracies

---

## 3. Methodology Overview

### 3.1 Three-Tier Testing Architecture

```
TIER 1: Single-Embedding Spectral Analysis
├── FFT magnitude/phase extraction
├── Periodogram analysis
└── Hypothesis: Spectral peaks at rational fractions

TIER 2: Cross-Embedding Spectral Coherence  
├── Cross-spectral density (CSD)
├── Magnitude-squared coherence (MSC)
└── Hypothesis: Coherence > 0.8 at semantic frequencies

TIER 3: Population-Level Spectral Statistics
├── Multi-model spectral comparison
├── Meta-analysis across vocabularies
└── Hypothesis: Universal spectral structure
```

### 3.2 Novel Techniques

**A. Complex Morlet Wavelet Transform:**
```python
# Continuous wavelet transform with complex Morlet
ψ(t) = π^(-1/4) e^(iω_0t) e^(-t²/2)
W_x(a,b) = ∫ x(t) (1/√a) ψ*((t-b)/a) dt
```
Detects time-scale phase structure in embedding sequences.

**B. Hilbert-Huang Transform:**
```python
# Empirical mode decomposition + Hilbert transform
IMFs = EMD(x)  # Intrinsic mode functions
Instantaneous_phase = arg(Hilbert(IMF))
```
Reveals instantaneous frequency/phase without stationarity assumptions.

**C. Bispectral Analysis:**
```python
# Detects quadratic phase coupling
B(f1, f2) = E[X(f1)X(f2)X*(f1+f2)]
```
Identifies phase-locked harmonic relationships.

**D. Spectral Granger Causality:**
```python
# Directed spectral influence
S_{x→y}(f) = ln(|S_y(f)| / |S_{y|x}(f)|)
```
Tests if semantic "cause" predicts embedding "effect" in frequency domain.

---

## 4. Step-by-Step Protocol

### PHASE 1: Data Preparation (Day 1)

**Step 1.1: Vocabulary Curation**
```python
vocabularies = {
    "semantic_field_1": ["king", "queen", "prince", "monarch", "royal", "crown", "throne"],
    "semantic_field_2": ["man", "woman", "child", "adult", "person", "human", "individual"],
    "semantic_field_3": ["big", "small", "large", "tiny", "huge", "minute", "enormous"],
    # ... 10 semantic fields, 50 words each
}
```
- Stratified sampling across semantic domains
- Balanced word frequencies
- Include control: random word sets with no semantic relations

**Step 1.2: Embedding Extraction**
```python
models = [
    "all-MiniLM-L6-v2",
    "bert-base-uncased", 
    "all-mpnet-base-v2",
    "glove-wiki-300",
    "word2vec-google-300"
]
```
- Extract embeddings for all vocabularies
- Normalize to unit sphere
- Generate 500+ samples per model

**Step 1.3: Ground Truth Construction**
- Human-annotated semantic similarity matrices
- WordNet path similarity
- ConceptNet relatedness scores
- Independent validation set (20% holdout)

### PHASE 2: Single-Embedding Analysis (Days 2-3)

**Test 2.1: FFT Periodicity Detection**

**Procedure:**
```python
def test_fft_periodicity(embedding):
    # Zero-pad to power of 2
    x = np.fft.fft(embedding, n=next_pow2(len(embedding)))
    
    # Compute power spectrum
    power = np.abs(x) ** 2
    freqs = np.fft.fftfreq(len(x))
    
    # Test for 8-octant periodicity
    expected_peaks = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]
    
    for peak_freq in expected_peaks:
        idx = np.argmin(np.abs(freqs - peak_freq))
        local_power = power[idx-2:idx+3]
        baseline = np.mean(power)
        
        # Peak detection with Bonferroni correction
        if np.max(local_power) > baseline * threshold:
            significant_peaks.append(peak_freq)
    
    return significant_peaks
```

**Statistical Test:**
- Null: Power spectrum is white noise (uniform)
- Alternative: Peaks at rational fractions (1/8, 1/4, 1/2)
- Test: Chi-square goodness-of-fit
- Threshold: χ² > critical value at α = 0.00001
- Bonferroni correction for multiple frequency tests

**Expected Result (H1):** Peaks at f = k/8 for k=1..7 with p < 0.00001

---

**Test 2.2: Autocorrelation Oscillation**

**Procedure:**
```python
def test_autocorr_oscillation(embedding, max_lag=384):
    # Compute autocorrelation
    autocorr = np.correlate(embedding, embedding, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize
    
    # Fit damped oscillator
    def damped_oscillator(t, A, γ, ω, φ):
        return A * np.exp(-γ*t) * np.cos(ω*t + φ)
    
    t = np.arange(len(autocorr))
    popt, _ = curve_fit(damped_oscillator, t, autocorr, 
                        p0=[1.0, 0.01, 2*np.pi/48, 0])
    
    # Test if oscillation frequency is 8e-related
    predicted_ω = 2 * np.pi / (384 / (8 * np.e))
    
    return np.abs(popt[2] - predicted_ω) < tolerance
```

**Statistical Test:**
- Null: Autocorrelation is exponential decay (Markov process)
- Alternative: Damped oscillation with ω ≈ 2π/(384/8e)
- Test: F-test comparing exponential vs. damped oscillator fit
- Threshold: F > F_crit at α = 0.00001

**Expected Result (H1):** Oscillation period T ≈ 384/(8e) ≈ 17.6 dimensions

---

**Test 2.3: Hilbert Phase Coherence**

**Procedure:**
```python
def test_hilbert_coherence(embeddings):
    phases = []
    for emb in embeddings:
        # Analytic signal via Hilbert transform
        analytic = signal.hilbert(emb)
        inst_phase = np.unwrap(np.angle(analytic))
        phases.append(inst_phase)
    
    # Test phase coherence across embeddings
    phase_matrix = np.array(phases)
    
    # Circular variance (lower = more coherent)
    R = np.abs(np.mean(np.exp(1j * phase_matrix), axis=0))
    circular_variance = 1 - R
    
    # Phase locking value
    plv = np.abs(np.mean(np.exp(1j * np.diff(phase_matrix, axis=1)), axis=0))
    
    return np.mean(plv), circular_variance
```

**Statistical Test:**
- Null: Phases are uniformly distributed (Rayleigh test)
- Alternative: Phase concentration (non-uniform)
- Test: Rayleigh test for non-uniformity
- Threshold: R > R_crit at α = 0.00001

**Expected Result (H1):** PLV > 0.7, rejecting uniformity at p < 0.00001

---

### PHASE 3: Cross-Embedding Coherence (Days 4-5)

**Test 3.1: Cross-Spectral Density Analysis**

**Procedure:**
```python
def test_cross_spectral(emb1, emb2):
    # Welch's method for CSD
    f, csd = signal.csd(emb1, emb2, fs=1.0, nperseg=128, 
                        noverlap=64, scaling='spectrum')
    
    # Magnitude-squared coherence
    f1, psd1 = signal.welch(emb1, fs=1.0, nperseg=128)
    f2, psd2 = signal.welch(emb2, fs=1.0, nperseg=128)
    
    coherence = np.abs(csd)**2 / (psd1 * psd2)
    
    # Test semantic vs random pairs
    return coherence
```

**Statistical Test:**
- Compare coherence for semantically related vs. unrelated word pairs
- Null: No difference in coherence distribution
- Alternative: Semantic pairs have higher coherence
- Test: Mann-Whitney U with exact permutation p-value
- Threshold: p < 0.00001 with effect size r > 0.5

**Expected Result (H1):** Semantic pairs: MSC = 0.85 ± 0.10; Random pairs: MSC = 0.12 ± 0.08

---

**Test 3.2: Granger Causality in Spectral Domain**

**Procedure:**
```python
def test_spectral_granger(emb_cause, emb_effect):
    # Vector autoregression in frequency domain
    mvar = MVGC(max_order=10)
    mvar.fit(np.vstack([emb_cause, emb_effect]).T)
    
    # Spectral Granger causality
    f = np.linspace(0, 0.5, 100)
    gc_spectrum = mvar.spectral_causality(0, 1, freqs=f)
    
    # Test if semantic direction predicts spectral flow
    return gc_spectrum
```

**Statistical Test:**
- Null: No directed spectral influence
- Alternative: Asymmetric spectral flow (semantic → embedding)
- Test: Parametric bootstrap with 10000 resamples
- Threshold: p < 0.00001 for asymmetry index

**Expected Result (H1):** G-causality asymmetry > 0.6 at semantic frequencies

---

**Test 3.3: Phase Synchronization Index**

**Procedure:**
```python
def test_phase_synchronization(embeddings_pairs):
    psi_values = []
    
    for emb1, emb2 in embeddings_pairs:
        # Bandpass filter around 8e frequency
        sos = signal.butter(4, [0.04, 0.06], btype='band', fs=1.0, output='sos')
        filtered1 = signal.sosfilt(sos, emb1)
        filtered2 = signal.sosfilt(sos, emb2)
        
        # Instantaneous phase
        phase1 = np.angle(signal.hilbert(filtered1))
        phase2 = np.angle(signal.hilbert(filtered2))
        
        # Phase difference
        delta_phi = phase1 - phase2
        
        # Phase synchronization index
        psi = np.abs(np.mean(np.exp(1j * delta_phi)))
        psi_values.append(psi)
    
    return psi_values
```

**Statistical Test:**
- Null: PSI = 1/√N (random phase difference)
- Alternative: PSI >> 1/√N (phase-locked)
- Test: One-sample t-test against theoretical null
- Threshold: t > t_crit(α=0.00001, df=n-1)

**Expected Result (H1):** PSI > 0.8 for semantic pairs vs. PSI ≈ 0.1 for random

---

### PHASE 4: Population-Level Meta-Analysis (Days 6-7)

**Test 4.1: Multi-Model Spectral Convergence**

**Procedure:**
```python
def test_multimodel_convergence(models_results):
    spectra = []
    for model_name, embeddings in models_results.items():
        avg_spectrum = np.mean([np.abs(np.fft.fft(e))**2 for e in embeddings], axis=0)
        spectra.append(avg_spectrum)
    
    # Test if spectra converge across models
    spectral_corr = np.corrcoef(spectra)
    
    return spectral_corr
```

**Statistical Test:**
- Null: Independent spectral structures per model
- Alternative: Universal spectral signature
- Test: Fisher's z-transform with Bonferroni correction
- Threshold: r > 0.9 for all model pairs at p < 0.00001

**Expected Result (H1):** Cross-model correlation > 0.95, indicating universal structure

---

**Test 4.2: Vocabulary Robustness**

**Procedure:**
```python
def test_vocabulary_robustness(vocab_sizes=[50, 100, 200, 500, 1000]):
    results = {}
    
    for size in vocab_sizes:
        # Random subsample
        subsamples = [random.sample(all_words, size) for _ in range(100)]
        
        # Compute spectral invariants per subsample
        invariants = []
        for vocab in subsamples:
            embs = get_embeddings(vocab)
            spec = compute_spectrum(embs)
            invariants.append(extract_invariants(spec))
        
        # Coefficient of variation
        cv = np.std(invariants) / np.mean(invariants)
        results[size] = cv
    
    return results
```

**Statistical Test:**
- Null: Spectral invariants vary with vocabulary
- Alternative: Invariant to vocabulary composition
- Test: Levene's test for homogeneity of variance
- Threshold: p > 0.1 (fail to reject homogeneity)

**Expected Result (H1):** CV < 5% for vocabularies > 200 words

---

**Test 4.3: Meta-Analytic Effect Size**

**Procedure:**
```python
def meta_analyze(results_list):
    # Fixed-effects meta-analysis
    effect_sizes = [r.effect_size for r in results_list]
    variances = [r.variance for r in results_list]
    
    # Inverse variance weighting
    weights = 1 / np.array(variances)
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_var = 1 / np.sum(weights)
    
    # Heterogeneity (I² statistic)
    q_stat = np.sum(weights * (effect_sizes - pooled_effect)**2)
    i_squared = max(0, (q_stat - (len(effect_sizes) - 1)) / q_stat * 100)
    
    return pooled_effect, pooled_var, i_squared
```

**Statistical Test:**
- Compute pooled effect size across all tests
- Heterogeneity I² < 50% indicates consistent effects
- Threshold: Pooled effect > 0.8 with 95% CI excluding 0

**Expected Result (H1):** r = 0.85 [0.82, 0.88], I² = 15%

---

## 5. Controls and Confounds

### 5.1 Negative Controls

**Control 1: Random Gaussian Vectors**
```python
def generate_null_embeddings(n_samples, dim):
    return np.random.randn(n_samples, dim)
```
- Should show white noise spectrum
- Should fail all periodicity tests
- Validates test specificity

**Control 2: Permutation Null**
```python
def permutation_test(embeddings, n_permutations=100000):
    null_distribution = []
    
    for _ in range(n_permutations):
        permuted = [np.random.permutation(e) for e in embeddings]
        stat = compute_test_statistic(permuted)
        null_distribution.append(stat)
    
    p_value = np.mean(np.array(null_distribution) >= observed_stat)
    return p_value
```
- Destroys any phase structure
- Should produce uniform p-values
- Validates statistical calibration

**Control 3: Phase-Scrambled Surrogates**
```python
def phase_scramble_surrogate(embedding):
    fft = np.fft.fft(embedding)
    magnitude = np.abs(fft)
    random_phase = np.random.uniform(0, 2*np.pi, len(fft))
    scrambled = magnitude * np.exp(1j * random_phase)
    return np.real(np.fft.ifft(scrambled))
```
- Preserves power spectrum, destroys phase coherence
- Tests if results depend on phase vs. power

### 5.2 Confound Detection

**Confound 1: Embedding Dimension Artifacts**
- **Issue:** Power-of-2 padding creates spectral leakage
- **Mitigation:** Use prime-length windows, zero-padding with Tukey window

**Confound 2: Training Data Frequency Bias**
- **Issue:** Word frequencies in training corpus create spectral peaks
- **Mitigation:** Test on controlled vocabularies with matched frequencies

**Confound 3: Positional Encoding Residuals**
- **Issue:** Transformer positional encodings have Fourier structure
- **Mitigation:** Use sentence-level averaging, test non-transformer models (Word2Vec, GloVe)

**Confound 4: Multiple Comparisons Inflation**
- **Issue:** Testing 384 frequencies inflates false positive rate
- **Mitigation:** Bonferroni correction: α = 0.00001/384 ≈ 2.6e-8

**Confound 5: Circular Statistics Bias**
- **Issue:** Phase wraparound creates artifacts
- **Mitigation:** Use circular statistics (Rayleigh, Watson-U²), unwrap phases

---

## 6. Irrefutable Proof Criteria

### 6.1 Necessary Conditions for Confirmation

For Q51 to be considered **ABSOLUTELY PROVEN**, ALL of the following must hold:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Spectral Peaks** | p < 10^-5 at f=k/8 | 8-fold periodicity from Q48-Q50 |
| **Cross-Model Agreement** | r > 0.95 | Universal structure across architectures |
| **Phase Coherence** | PLV > 0.7 | Complex phase organization |
| **Semantic Coherence** | MSC > 0.8 vs 0.1 random | Meaning maps to spectral structure |
| **Granger Asymmetry** | p < 10^-5 for semantic flow | Causal direction semantic → embedding |
| **Vocabulary Invariance** | CV < 5% | Structure is fundamental, not artifact |
| **Null Control Failure** | p > 0.01 for all nulls | Test specificity confirmed |
| **Meta-Analytic Effect** | r > 0.8 [0.75, 0.85] | Consistent across all tests |
| **Bonferroni Survival** | All p < 2.6×10^-8 | Survives 384-fold correction |
| **Effect Size** | Cohen's d > 2.0 | Large, practically significant effect |

### 6.2 Sufficient Conditions Hierarchy

**Level 1: Preliminary (p < 0.001)**
- Single-test significance
- Suggests further investigation warranted

**Level 2: Probable (p < 0.00001 + Bonferroni)**
- All primary tests significant
- Cross-validation across models
- **Status: Q51 likely true**

**Level 3: Definitive (All criteria met)**
- All 10 criteria satisfied
- Multiple independent replications
- Published negative controls
- **Status: Q51 ABSOLUTELY PROVEN**

**Level 4: Irrefutable (Theory confirmed)**
- Level 3 + mechanistic explanation
- Predictive power demonstrated
- Alternative theories falsified
- **Status: Complex semiotic space established as scientific fact**

### 6.3 Falsification Criteria

If ANY of the following occur, Q51 is **FALSIFIED**:

1. Random embeddings pass any test (false positive)
2. Spectral peaks don't align with 8-octant structure
3. Cross-model correlation < 0.5 (architecture-dependent artifact)
4. Bonferroni-corrected p > 0.01 for any primary test
5. Effect size Cohen's d < 0.5 (trivial effect)
6. Phase scrambling doesn't destroy effect (power-only artifact)

---

## 7. Expected Outcomes

### 7.1 Scenario A: Absolute Confirmation

**Expected Pattern:**
```
Spectral Peaks: [1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8] ✓
Cross-Model r: 0.97 [0.96, 0.98] ✓
Phase Locking Value: 0.82 ± 0.05 ✓
Semantic Coherence: 0.88 vs 0.09 (random) ✓
Meta-Analysis: r = 0.87 [0.84, 0.90] ✓
```

**Interpretation:** 
- Complex semiotic space is proven
- Real embeddings are projections
- Phase information encoded in spectral structure
- Universal across models

### 7.2 Scenario B: Partial Confirmation

**Expected Pattern:**
```
Spectral Peaks: Detected but weak (p < 0.001) ⚠
Cross-Model r: 0.72 [0.65, 0.78] ⚠
Phase Locking Value: 0.45 ± 0.12 ⚠
```

**Interpretation:**
- Weak complex structure exists
- Mixed real/complex signature
- Architecture-dependent effects
- May indicate training artifacts rather than fundamental structure

### 7.3 Scenario C: Falsification

**Expected Pattern:**
```
Spectral Peaks: Uniform distribution ✗
Cross-Model r: 0.31 [0.15, 0.47] ✗
Phase Locking Value: 0.12 ± 0.08 (≈random) ✗
Random embeddings pass tests ✗
```

**Interpretation:**
- No complex structure detected
- Real embeddings are truly real-valued
- Previous results were geometric artifacts
- Q51 hypothesis REJECTED

---

## 8. Statistical Validation Criteria

### 8.1 P-Value Thresholds

| Test Type | Raw Threshold | Bonferroni-Corrected | Minimum Effect Size |
|-----------|--------------|---------------------|---------------------|
| Primary | p < 10^-5 | p < 2.6×10^-8 | d > 1.5 |
| Secondary | p < 10^-4 | p < 2.6×10^-7 | d > 1.0 |
| Exploratory | p < 0.001 | - | d > 0.8 |
| Control | p > 0.01 | - | d < 0.2 |

### 8.2 Power Analysis

**Required Sample Sizes (for 99% power at α=10^-5):**

| Test | Effect Size | Minimum N |
|------|------------|-----------|
| Spectral Peak | d = 2.0 | 25 embeddings |
| Phase Coherence | r = 0.7 | 50 pairs |
| Cross-Model | r = 0.9 | 5 models |
| Meta-Analysis | r = 0.8 | 20 tests |

**Planned Sample:**
- 1000 embeddings per model
- 500 semantic pairs
- 5 embedding models
- 10 vocabulary sets

**Total Statistical Power:** > 99.9% for detecting d > 1.5

### 8.3 Multiple Testing Corrections

**Family-Wise Error Rate Control:**
```python
# Holm-Bonferroni procedure (more powerful than Bonferroni)
def holm_bonferroni(p_values, alpha=0.00001):
    sorted_p = np.sort(p_values)
    n = len(p_values)
    
    for i, p in enumerate(sorted_p):
        if p > alpha / (n - i):
            return False  # Fail to reject
    return True  # All significant
```

**False Discovery Rate (FDR) Control:**
```python
# Benjamini-Hochberg for exploratory tests
def benjamini_hochberg(p_values, fdr=0.01):
    sorted_p = np.sort(p_values)
    n = len(p_values)
    
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) / n * fdr:
            return True
    return False
```

---

## 9. Code Structure Outline

### 9.1 Repository Layout

```
phase_4_absolute_proof/
├── fourier_approach/
│   ├── RESEARCH_PROPOSAL.md          # This document
│   ├── IMPLEMENTATION_REPORT.md      # Post-execution results
│   ├── src/
│   │   ├── __init__.py
│   │   ├── spectral_analysis.py      # Core FFT/periodogram
│   │   ├── coherence.py              # Cross-spectral methods
│   │   ├── hilbert_analysis.py       # Instantaneous phase
│   │   ├── wavelet_transform.py      # CWT with complex Morlet
│   │   ├── bispectral.py             # Quadratic phase coupling
│   │   ├── granger_causality.py      # Spectral GC
│   │   ├── meta_analysis.py          # Pooled effect sizes
│   │   ├── controls.py               # Null models & surrogates
│   │   └── statistics.py             # Circular stats, corrections
│   ├── tests/
│   │   ├── test_fft_periodicity.py   # Test 2.1
│   │   ├── test_autocorr.py          # Test 2.2
│   │   ├── test_hilbert_coherence.py # Test 2.3
│   │   ├── test_cross_spectral.py    # Test 3.1
│   │   ├── test_granger.py           # Test 3.2
│   │   ├── test_phase_sync.py        # Test 3.3
│   │   ├── test_multimodel.py        # Test 4.1
│   │   ├── test_vocabulary.py        # Test 4.2
│   │   ├── test_meta_analysis.py     # Test 4.3
│   │   └── test_controls.py          # All negative controls
│   ├── fixtures/
│   │   ├── input/
│   │   │   ├── semantic_fields.json  # Curated vocabularies
│   │   │   └── ground_truth.json     # Human annotations
│   │   └── expected/
│   │       ├── null_spectrum.json    # Random Gaussian
│   │       └── periodic_spectrum.json # Theoretical complex
│   ├── results/
│   │   ├── phase1_preparation/       # Data validation logs
│   │   ├── phase2_single/            # Single-embedding results
│   │   ├── phase3_cross/             # Cross-embedding results
│   │   ├── phase4_meta/              # Population-level results
│   │   └── final_report/             # Aggregated analysis
│   ├── notebooks/
│   │   ├── 01_exploratory_eda.ipynb  # Initial data exploration
│   │   ├── 02_spectral_viz.ipynb     # Spectrum visualization
│   │   ├── 03_phase_analysis.ipynb   # Phase dynamics
│   │   └── 04_final_synthesis.ipynb  # Result integration
│   └── config/
│       ├── models.yaml               # Model configurations
│       ├── thresholds.yaml           # Statistical thresholds
│       └── vocabularies.yaml         # Word list definitions
```

### 9.2 Core Algorithm Templates

**Template 1: Multi-Taper Spectral Estimation**
```python
"""
Thomson's multi-taper method for optimal spectral estimation.
Reduces variance while maintaining frequency resolution.
"""
from spectrum import dpss, pmtm

def multitaper_spectrum(embedding, nw=2.5, k=5):
    """
    Compute adaptive multi-taper spectrum.
    
    Args:
        embedding: 1D array of embedding values
        nw: Time-bandwidth product (2.5 = standard)
        k: Number of Slepian tapers (2*nw - 1 typically)
    
    Returns:
        freqs: Frequency bins
        power: Adaptive weighted spectrum
        conf_interval: 95% confidence intervals
    """
    # Generate discrete prolate spheroidal sequences
    tapers, eigenvalues = dpss(len(embedding), nw, k)
    
    # Compute tapered spectra
    spectra = []
    for taper in tapers:
        tapered = embedding * taper
        spectrum = np.abs(np.fft.fft(tapered))**2
        spectra.append(spectrum)
    
    # Adaptive weighting
    weights = eigenvalues / np.sum(eigenvalues)
    adaptive_spectrum = np.average(spectra, axis=0, weights=weights)
    
    # Chi-square confidence intervals
    df = 2 * k  # degrees of freedom
    lower = adaptive_spectrum * df / chi2.ppf(0.975, df)
    upper = adaptive_spectrum * df / chi2.ppf(0.025, df)
    
    freqs = np.fft.fftfreq(len(embedding))
    return freqs, adaptive_spectrum, (lower, upper)
```

**Template 2: Complex Wavelet Phase Extraction**
```python
"""
Continuous wavelet transform with complex Morlet wavelet.
Optimal for detecting time-scale phase structure.
"""
import pywt

def complex_morlet_cwt(embedding, scales=None, w0=6):
    """
    Perform CWT with complex Morlet wavelet.
    
    Args:
        embedding: 1D signal (embedding components)
        scales: Wavelet scales to analyze (default: 1-128)
        w0: Central frequency (6 = good balance)
    
    Returns:
        coefficients: Complex wavelet coefficients (scales × time)
        scales: Analyzed scales
        phase: Instantaneous phase at each scale
        power: Wavelet power spectrum
    """
    if scales is None:
        scales = np.arange(1, 129)
    
    # Complex Morlet wavelet
    wavelet = f'cmor{w0}-1.0'
    
    # Continuous wavelet transform
    coefficients, freqs = pywt.cwt(embedding, scales, wavelet)
    
    # Extract phase and power
    phase = np.angle(coefficients)
    power = np.abs(coefficients)**2
    
    return coefficients, scales, freqs, phase, power
```

**Template 3: Circular Statistical Tests**
```python
"""
Circular statistics for phase analysis.
Handles 2π periodicity correctly.
"""
from scipy import stats

def circular_mean(angles):
    """Compute circular mean of angles in radians."""
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    return np.arctan2(sin_mean, cos_mean)

def circular_variance(angles):
    """Compute circular variance (0 = uniform, 1 = concentrated)."""
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    return 1 - R

def rayleigh_test(angles):
    """
    Rayleigh test for uniformity of circular data.
    
    Returns:
        R: Mean resultant length
        p_value: Probability of uniformity
    """
    n = len(angles)
    R = np.sqrt(np.sum(np.sin(angles))**2 + np.sum(np.cos(angles))**2) / n
    
    # Rayleigh statistic
    Z = n * R**2
    
    # P-value approximation (for large n)
    p_value = np.exp(-Z)
    
    return R, p_value

def watson_u2_test(angles1, angles2):
    """
    Watson's U² test for equality of two circular distributions.
    Non-parametric, handles multi-modality.
    """
    # Combine and rank
    combined = np.concatenate([angles1, angles2])
    ranks = stats.rankdata(combined)
    
    n1, n2 = len(angles1), len(angles2)
    R1 = np.sum(ranks[:n1])
    
    # U² statistic
    U2 = (R1 - n1*(n1+n2+1)/2)**2 / (n1*n2*(n1+n2+1)/12)
    
    # P-value from approximation
    p_value = 1 - stats.chi2.cdf(U2, df=2)
    
    return U2, p_value
```

**Template 4: Bispectral Analysis**
```python
"""
Bispectral analysis for detecting quadratic phase coupling.
Identifies harmonic relationships between frequencies.
"""
from scipy.signal import correlate

def bispectrum(embedding, nfft=None):
    """
    Compute direct bispectrum estimate.
    
    Args:
        embedding: 1D signal
        nfft: FFT length (default: next power of 2)
    
    Returns:
        B: Bispectrum matrix (f1 × f2)
        freqs: Frequency axes
    """
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(len(embedding))))
    
    # FFT
    X = np.fft.fft(embedding, nfft)
    X = X[:nfft//2]  # Keep positive frequencies only
    
    # Initialize bispectrum matrix
    n_freq = len(X)
    B = np.zeros((n_freq, n_freq), dtype=complex)
    
    # Compute B(f1, f2) = E[X(f1) X(f2) X*(f1+f2)]
    for i in range(n_freq):
        for j in range(n_freq):
            if i + j < n_freq:
                B[i, j] = X[i] * X[j] * np.conj(X[i+j])
    
    # Normalize to bicoherence
    for i in range(n_freq):
        for j in range(n_freq):
            if i + j < n_freq:
                denom = np.abs(X[i])**2 * np.abs(X[j])**2 * np.abs(X[i+j])**2
                if denom > 0:
                    B[i, j] = np.abs(B[i, j])**2 / denom
    
    freqs = np.fft.fftfreq(nfft)[:nfft//2]
    return B, freqs
```

---

## 10. Implementation Timeline

### Week 1: Foundation (Days 1-7)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Vocabulary curation + embedding extraction | `semantic_fields.json` |
| 2 | Single-embedding FFT pipeline | `spectral_analysis.py` |
| 3 | Autocorrelation + Hilbert tests | `test_autocorr.py`, `test_hilbert.py` |
| 4 | Cross-spectral methods | `coherence.py` |
| 5 | Granger causality + phase sync | `test_granger.py`, `test_phase_sync.py` |
| 6 | Control implementations | `controls.py` |
| 7 | Phase 1-2 integration test | Phase 2 results directory |

### Week 2: Integration (Days 8-14)

| Day | Task | Deliverable |
|-----|------|-------------|
| 8 | Multi-model framework | `test_multimodel.py` |
| 9 | Vocabulary robustness | `test_vocabulary.py` |
| 10 | Meta-analysis pipeline | `meta_analysis.py` |
| 11 | Full system integration | End-to-end test pass |
| 12 | Large-scale execution | 1000+ embeddings processed |
| 13 | Statistical validation | All p-values computed |
| 14 | Preliminary analysis | Phase 3 results directory |

### Week 3: Validation (Days 15-21)

| Day | Task | Deliverable |
|-----|------|-------------|
| 15 | Control verification | All nulls pass |
| 16 | Sensitivity analysis | Robustness checks |
| 17 | Cross-validation | Independent replication |
| 18 | Visualization | Publication-ready plots |
| 19 | Report drafting | `IMPLEMENTATION_REPORT.md` |
| 20 | Peer review | Internal validation |
| 21 | Final submission | Complete result package |

---

## 11. Risk Assessment

### 11.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Computational cost** | High | Medium | Cloud GPU, parallel processing |
| **Memory constraints** | Medium | Medium | Streaming algorithms, batching |
| **Numerical precision** | Low | High | Multi-precision arithmetic, validation |
| **Library dependencies** | Medium | Low | Docker container, version pinning |

### 11.2 Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **No spectral structure found** | Medium | Critical | Pre-registered analysis plan |
| **Inconclusive results** | Medium | High | Bayesian analysis, Bayes factors |
| **Confound contamination** | High | Medium | Extensive control battery |
| **Replication failure** | Low | Critical | Multi-model, multi-vocab design |

### 11.3 Contingency Plans

**If spectral peaks are weak:**
- Increase sample size to 10,000 embeddings
- Use Bayesian spectral estimation (more sensitive)
- Apply signal processing (filtering, denoising)

**If controls fail:**
- Investigate specific failure mode
- Add diagnostic tests
- Adjust thresholds if justified

**If results are mixed:**
- Conduct Bayesian model comparison
- Compute Bayes factors
- Report uncertainty quantitatively

---

## 12. Success Metrics

### 12.1 Quantitative Metrics

| Metric | Target | Acceptable | Unacceptable |
|--------|--------|------------|--------------|
| **Primary test p-values** | < 10^-8 | < 10^-5 | > 10^-4 |
| **Effect size (Cohen's d)** | > 2.0 | > 1.0 | < 0.5 |
| **Cross-model correlation** | > 0.95 | > 0.80 | < 0.50 |
| **Control pass rate** | 100% | > 95% | < 90% |
| **Sample size** | 1000 | > 500 | < 200 |
| **Reproducibility** | 100% | > 90% | < 75% |

### 12.2 Qualitative Criteria

- **Scientific Rigor:** Pre-registered, blinded where possible
- **Transparency:** Full code, data, and analysis public
- **Reproducibility:** Independent replication within 5% tolerance
- **Interpretability:** Results explainable to non-experts
- **Impact:** Changes understanding of embedding structure

---

## 13. Conclusion

This Fourier/spectral methodology offers a **fundamentally new approach** to Q51 that:

1. **Detects hidden structure** invisible to geometric methods
2. **Provides absolute thresholds** (p < 0.00001) for certainty
3. **Includes comprehensive controls** against all known confounds
4. **Enables irrefutable proof** through convergent multi-test evidence
5. **Sets new standard** for rigorous embedding analysis

**The core insight:** If embeddings are projections of complex meaning, the projection process leaves **spectral fingerprints** in the frequency domain. These fingerprints are:
- **Unique to complex structure** (no real process produces them)
- **Recoverable via Fourier analysis** (even after projection)
- **Testable with absolute certainty** (via strict statistical thresholds)

**Expected Outcome:** Either Q51 will be **ABSOLUTELY PROVEN** with p < 10^-8 across all tests, or it will be **DEFINITIVELY FALSIFIED** with power > 99%. No ambiguous middle ground—this methodology forces a clear verdict.

---

## Appendix A: Mathematical Derivations

### A.1 Projection-Induced Spectral Asymmetry

Given z(t) = x(t) + iy(t) with Fourier transform Z(f) = X(f) + iY(f)

The real projection x(t) has Fourier transform:
X_proj(f) = 1/2[Z(f) + Z*(-f)]

For complex z(t) with unilateral spectrum:
Z(f) = 0 for f < 0

Thus:
X_proj(f) = 1/2 Z(f) for f > 0
X_proj(-f) = 1/2 Z*(f)

**Result:** The projection creates **Hermitian symmetry** with phase information encoded in the relationship between positive and negative frequencies.

### A.2 Octant Periodicity from 8e Symmetry

From Q48-Q50: Df × α = 8e ≈ 21.76

In Fourier domain:
- Scaling symmetry: f → f/2 (doubling dimension)
- Phase periodicity: θ → θ + 2π/8 (8-octant structure)

The 8-fold symmetry creates spectral peaks at:
f_k = k × f_0 / 8 for k = 1, 2, ..., 7

Where f_0 is the fundamental frequency related to embedding dimension d=384.

### A.3 Berry Phase as Spectral Holonomy

For a closed loop C in parameter space:
γ_B = ∮_C A where A = i⟨ψ|∇|ψ⟩ = Im[⟨ψ|∇ψ⟩]

In spectral domain, this manifests as:
- Phase accumulation: Δφ = ∮ ∇φ · dl
- Singularities: Vortices where |ψ| = 0
- Quantization: γ_B = 2πn for integer n

**Detectable as:** 2π phase jumps in spectral phase plots.

---

## Appendix B: Software Requirements

### B.1 Required Libraries

```
numpy >= 1.24.0
scipy >= 1.10.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
pywavelets >= 1.4.0
spectrum >= 0.8.0
mne >= 1.4.0
nitime >= 0.9.0
statsmodels >= 0.14.0
pingouin >= 0.5.3
scikit-learn >= 1.3.0
sentence-transformers >= 2.2.0
transformers >= 4.30.0
torch >= 2.0.0
gensim >= 4.3.0
nltk >= 3.8.0
```

### B.2 Hardware Requirements

**Minimum:**
- 16GB RAM
- 4 CPU cores
- 100GB storage

**Recommended:**
- 64GB RAM
- 8+ CPU cores / 1 GPU
- 500GB SSD storage

---

## References

1. Thomson, D.J. (1982). Spectrum estimation and harmonic analysis. *Proc. IEEE*, 70(9), 1055-1096.
2. Mallat, S. (2008). *A Wavelet Tour of Signal Processing*. Academic Press.
3. Faes, L., & Nollo, G. (2011). Bivariate frequency domain measures of linear coupling. *Phil. Trans. R. Soc. A*, 367(1887), 1263-1281.
4. Lachaux, J.P., et al. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4), 194-208.
5. Granger, C.W.J. (1969). Investigating causal relations by econometric models. *Econometrica*, 37(3), 424-438.
6. Fisher, N.I. (1993). *Statistical Analysis of Circular Data*. Cambridge University Press.
7. Efron, B., & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

---

**Document Status:** PROPOSED  
**Version:** 1.0  
**Next Step:** Implementation upon approval  
**Reviewers:** [Pending]  
**Estimated Duration:** 21 days  
**Resource Requirements:** [See Appendix B]

---

*This research proposal establishes a rigorous, statistically bulletproof methodology for definitively answering Q51. The Fourier/spectral approach provides capabilities impossible with spatial-domain methods, potentially enabling absolute certainty about the complex nature of semantic space.*
