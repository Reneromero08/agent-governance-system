# Provenance Analysis: R_mi ~2x Decoherence Prediction

**Date:** 2026-01-30
**Investigator:** Claude Opus 4.5
**Status:** COMPLETE - **THE PREDICTION IS AD-HOC, NOT DERIVED**

---

## Executive Summary

The "R_mi increases ~2x during decoherence" prediction was **NOT** derived from the core formula R = (E/grad_S) * sigma^Df. It was:

1. **First observed** in a single simulation with specific parameters
2. **Generalized** to a "universal" prediction without theoretical justification
3. **Post-hoc labeled** as a "pre-registered prediction" despite arising from simulation

This analysis documents the full provenance chain and provides an honest assessment.

---

## 1. The Derivation Chain (What SHOULD Exist)

### Claimed Relationship

The Q54 framework claims:

```
R_mi = (E_mi / grad_mi) * sigma^Df

Where:
  E_mi = average mutual information between system and fragments
  grad_mi = standard deviation of MI across fragments
  Df = log(n_fragments + 1)
  sigma = 0.5 (constant)
```

### What a Proper Derivation Would Look Like

To derive "R_mi increases ~2x during decoherence" from first principles, we would need:

1. **Functional form of E_mi(t):** How does average MI evolve during decoherence?
2. **Functional form of grad_mi(t):** How does MI dispersion evolve?
3. **Analytical solution:** Show that lim(t->infinity) R_mi / R_mi(0) = 2

**Such a derivation does NOT exist in the codebase.**

---

## 2. What Actually Happened (Chronological Trace)

### Stage 1: Original Hypothesis (Qualitative)

From `q54_energy_spiral_matter.md` (lines 122-125):

```
### Prediction 3: Phase Lock Precedes Observation
The spiral must crystallize before "matter" is observed.
**Test:** In decoherence experiments, R should spike before classical behavior emerges.
```

**Note:** This was a *qualitative* prediction: "R should spike." No number was stated.

### Stage 2: Simulation Created

File: `test_c_zurek_data.py`

Parameters used (lines 57-82):
```python
class DecoherenceParams:
    coupling = 0.5
    n_env_modes = 6
    t_max = 5.0
    n_timesteps = 100
```

The R_mi metric was *invented* for this test (lines 329-380):
```python
def compute_R_mi(state, n_total, sigma=0.5):
    """R based on Mutual Information - the correct metric for QD."""
    # ... implementation
    return (E_mi / grad_mi) * (sigma ** Df)
```

### Stage 3: First Results

From `test_c_zurek_results.json`:
```json
{
  "R_before_transition": 8.152438805423413,
  "R_after_transition": 16.804886410683356
}
```

The ratio: 16.80 / 8.15 = **2.06x**

**CRITICAL OBSERVATION:** The "2.0" value was FIRST SEEN HERE, not derived beforehand.

### Stage 4: Statistical Analysis

File: `test_c_statistical.py` ran 15 simulations with *varied parameters*:
- coupling varied 0.45-0.55
- 8% measurement noise added

Results from `test_c_statistical_results.json`:
```json
{
  "bootstrap": {
    "point_estimate": 2.198109414414597,
    "ci_95_lower": 2.07912140128516,
    "ci_95_upper": 2.318112463951779
  },
  "monte_carlo": {
    "mean_ratio": 2.1198357223395243,
    "min_ratio": 1.1268397753196666,
    "max_ratio": 5.389148679748427
  }
}
```

**Note the Monte Carlo range: 1.13x to 5.39x.** The ratio varies dramatically with parameters.

### Stage 5: Post-Hoc "Universality" Claim

From `PRE_REGISTRATION.md` (lines 80-97):

```markdown
## Prediction 2: R_mi Decoherence Spike (UNIVERSAL)

Point Estimate: 2.0
Standard Error: +/- 0.3
95% Confidence Interval: [1.4, 2.6]

UNIVERSALITY CLAIM: This ratio should be approximately 2.0 across ALL decoherence
experiments regardless of specific physical system
```

**PROBLEM:** The "universality claim" was made AFTER seeing the simulation results, not before.

### Stage 6: External Validation (Partial Failure)

From `INDEX.md` (lines 183-184):
```
| R_mi decoherence spike | 2.0 +/- 0.3 | 1.93 (frag 1) | Zhu et al. 2025 | **PARTIAL PASS** |
| R_mi universality | Universal 2x | Varies 1.3-3.7 | Same | **WEAKENED** |
```

Real data showed the ratio varies from 1.3 to 3.7 depending on fragment size - falsifying the universality claim.

---

## 3. Answering the Specific Questions

### Question 1: Was this derived from R = (E/grad_S) * sigma^Df?

**NO.** The formula predicts R_mi will *change* during decoherence (because E_mi increases and grad_mi may stay low), but it does NOT predict the ratio will be 2.0.

The formula structure is:
```
R_mi = (E_mi / grad_mi) * sigma^Df
```

For the ratio to be universal at 2.0, we would need:
```
R_after / R_before = (E_after/grad_after) / (E_before/grad_before) * sigma^(Df_after - Df_before) = 2.0
```

This requires specific relationships between E_mi, grad_mi, and Df evolution that are NOT derived anywhere.

### Question 2: What does the formula predict should depend on?

The formula predicts R_mi ratio should depend on:

| Parameter | In Formula? | Effect on Ratio |
|-----------|-------------|-----------------|
| Fragment size (via Df) | YES | Df = log(n_env + 1), larger env = smaller sigma^Df |
| Coupling strength | Implicit in E_mi | Stronger coupling = faster MI growth |
| Environment size | YES | More fragments = more averaging in E_mi |
| Initial state | Implicit | Different pure states give different dynamics |
| sigma value | YES | sigma^Df directly scales result |

**The formula predicts the ratio should NOT be universal - it should depend on all these parameters.**

The Monte Carlo results confirm this:
- Range: 1.13x to 5.39x
- Different couplings, environment sizes, and initial states give different ratios

### Question 3: What the simulation actually showed

The original simulation used ONE specific setup:
- n_env = 6 qubits
- coupling = 0.5
- sigma = 0.5
- Initial state: |+>

This gave 2.06x.

**Generalization to "universal 2x" was not justified.** When parameters varied, the ratio varied from 1.13x to 5.39x.

### Question 4: Honest Classification

The "R_mi increases ~2x" claim is:

| Classification | Verdict | Justification |
|----------------|---------|---------------|
| Derived from first principles? | **NO** | No analytical derivation exists |
| Empirical observation from simulation? | **YES** | First seen in test_c results |
| Universal prediction? | **NO** | Varies 1.1-5.4x across parameters |
| Post-hoc rationalization? | **PARTIALLY** | The quantitative "2x" was stated after simulation |

**Honest characterization:** "R_mi increases during decoherence" is qualitatively supported. The specific value "~2x" is an empirical observation from one simulation configuration, not a derived prediction.

---

## 4. What the Formula DOES Predict (Properly)

### Prediction A: R_mi Increases During Decoherence (QUALITATIVE)

**Derivation:**
```
R_mi = (E_mi / grad_mi) * sigma^Df

During decoherence:
1. E_mi increases (fragments gain MI with system)
2. grad_mi stays low (fragments gain similar MI - consensus)
3. Therefore: R_mi increases

Prediction: R_after > R_before
```

**This is properly derived** from the formula structure.

### Prediction B: Ratio Depends on Fragment Size

**Derivation:**
```
R_mi = (E_mi / grad_mi) * sigma^Df

Where: Df = log(n_env + 1)

If sigma < 1 (which it is, sigma = 0.5):
  - More fragments = larger Df = smaller sigma^Df
  - Ratio should vary systematically with fragment count
```

**This is also properly derived** - and confirmed by external validation showing ratio varies 1.3-3.7 with fragment size.

### Prediction C: No Universal Ratio

**Derived conclusion:** The formula does NOT predict a universal ratio. It predicts R_mi increases, with the magnitude depending on coupling, environment size, and initial state.

---

## 5. The Pre-Registration Problem

The file `PRE_REGISTRATION.md` claims:

> "Pre-Registration Date: 2026-01-30T12:00:00Z"
>
> "These predictions are documented BEFORE testing against real experimental data from external sources."

This is technically true - the predictions were stated before external validation. But:

1. The "2.0 +/- 0.3" value came from internal simulations
2. The simulations were run BEFORE the pre-registration
3. The "prediction" is therefore a post-diction of simulation results, not a theoretical prediction

**This is not proper pre-registration practice.** True pre-registration would require stating the prediction BEFORE any simulation or calculation.

---

## 6. Recommendations

### 6.1 Update the Prediction Statement

**Current (incorrect):**
```
Prediction: R_mi ratio = 2.0 +/- 0.3 (universal)
```

**Correct:**
```
Prediction: R_mi increases during decoherence (qualitative)
Observation: In specific simulations (n_env=6, coupling=0.5), ratio ~ 2.0
Caveat: Ratio varies 1.1-5.4x with parameters; not universal
```

### 6.2 Add Derivation or Acknowledge Gap

Either:
1. **Derive** why the ratio should be ~2 from the formula structure
2. **Acknowledge** that 2.0 is empirical, not derived

### 6.3 Update INDEX.md Status

Current entry:
```
| R_mi decoherence spike | 2.0 +/- 0.3 | 1.93 (frag 1) | PARTIAL PASS |
| R_mi universality | Universal 2x | Varies 1.3-3.7 | WEAKENED |
```

Should note that "2.0" was not a theoretical prediction.

### 6.4 Scientific Honesty

The Q54 framework makes VALID predictions:
- R_mi increases during decoherence (qualitative) - **DERIVED**
- Phase lock correlates with binding energy - **CONFIRMED (r=0.999)**
- Standing waves show more inertia - **PARTIALLY TESTED**

The "universal 2x" claim is the weakest part and should be downgraded.

---

## 7. Conclusion

### Summary Table

| Claim | Status | Evidence |
|-------|--------|----------|
| "R_mi increases during decoherence" | VALID (derived) | Follows from formula structure |
| "R_mi increases ~2x" | EMPIRICAL | First observed in simulation |
| "2x is universal" | FALSIFIED | External data shows 1.3-3.7x range |
| "2x was pre-registered" | MISLEADING | Value came from prior simulations |

### Final Verdict

**The "R_mi ~2x" prediction is ad-hoc, not derived.**

It was:
1. First observed in a specific simulation configuration
2. Generalized to "universal" without derivation
3. Falsified when tested with varied parameters

The qualitative prediction "R_mi increases during decoherence" remains valid and is properly derived from the formula. The specific numerical prediction should be retracted or explicitly labeled as empirical observation.

**Scientific integrity requires distinguishing between:**
- Predictions derived from theory (strong)
- Observations from simulations (medium)
- Post-hoc generalizations (weak)

The 2x claim is the third category.

---

## References

### Files Analyzed

1. `q54_energy_spiral_matter.md` - Original hypothesis document
2. `test_c_zurek_data.py` - Simulation implementation
3. `test_c_zurek_results.json` - First simulation results
4. `test_c_statistical.py` - Statistical analysis code
5. `test_c_statistical_results.json` - Monte Carlo results
6. `AUDIT_test_c.md` - Post-analysis audit
7. `PRE_REGISTRATION.md` - Claimed pre-registration
8. `INDEX.md` - External validation summary
9. `INVESTIGATION_SUMMARY.md` - Combined test summary

### Key Evidence Locations

- First "2.06x" observation: `test_c_zurek_results.json` lines 13-14
- Monte Carlo variance: `test_c_statistical_results.json` lines 85-87
- External validation failure: `INDEX.md` lines 183-184

---

*This analysis was conducted with scientific rigor. The framework has genuine strengths; honest acknowledgment of weaknesses strengthens rather than undermines it.*

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
