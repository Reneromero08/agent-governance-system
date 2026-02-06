# PROPER DERIVATION: Testing R = (E / grad_S) * sigma^Df

**Date:** 2026-01-30
**Status:** CRITICAL ANALYSIS
**Goal:** Derive a testable prediction FROM the formula, not validate a simulation

---

## The Problem We Must Solve

Previous tests (A, B, C) share a fatal flaw: they **implement** the formula in code, run it, and then claim the output is a "prediction." This is circular. The formula produces whatever the code says it produces.

A proper test requires:
1. OPERATIONAL definitions of each term for a specific physical system
2. ALGEBRAIC derivation of what the formula predicts
3. COMPARISON to an independent prediction (e.g., Zurek's Quantum Darwinism)
4. IDENTIFICATION of where predictions differ (if anywhere)

---

## PART 1: Operational Definitions for Decoherence

Consider a qubit system S coupled to an environment E of N qubits.

### 1.1 The Physical Setup

**Initial state:**
```
|psi_0> = (|0> + |1>)/sqrt(2) (x) |0>^(x)N
```

**Hamiltonian (CNOT-like interaction):**
```
H = sum_{k=1}^{N} g_k * sigma_z^S * sigma_x^{E_k}
```

**Evolution:** Pure unitary (no Lindblad terms)

### 1.2 What Happens Physically

At t = 0:
- System is in superposition (|0> + |1>)/sqrt(2)
- Environment is in product state |0...0>
- Total state is separable: rho_SE = rho_S (x) rho_E

At t = t_dec (decoherence time, roughly pi/(2*g*N)):
- System has become entangled with environment
- Total state is approximately: (|0>|0...0> + |1>|1...1>)/sqrt(2)
- Tracing over E: rho_S -> diagonal (mixed state)
- Each environment fragment Ek contains information about S

### 1.3 Operational Definitions of R-Formula Terms

For this decoherence scenario, I will now DEFINE each term operationally:

#### E (Essence/Energy):
**Definition:** E = I(S:Ek) / S(rho_S), the normalized mutual information between system and a single environment fragment.

- I(S:Ek) = S(rho_S) + S(rho_Ek) - S(rho_S,Ek)
- At t=0: I(S:Ek) = 0 (no correlation), so E = 0
- At t=t_dec: I(S:Ek) -> S(rho_S) (full correlation), so E -> 1

#### grad_S (Gradient of Action/Entropy):
**Definition:** grad_S = std(I(S:Ek)) over all fragments k, the dispersion of mutual information across fragments.

- At t=0: All I(S:Ek) = 0, so grad_S -> epsilon (small regularization)
- At t=t_dec: All I(S:Ek) -> S(rho_S) uniformly, so grad_S -> epsilon

#### sigma (Scale Parameter):
**Definition:** sigma = exp(i * phi), where phi = avg phase of system.

For simplicity, take sigma = 1/2 (a fixed scaling constant). This means we are NOT testing sigma as a dynamical variable in this derivation.

#### Df (Fractal/Effective Dimension):
**Definition:** Df = log(N+1), where N is the number of environment fragments.

This captures the "redundancy dimension" - how many independent copies of information exist.

---

## PART 2: Deriving R_before and R_after

### 2.1 R at t = 0 (Before Decoherence)

```
E_before = I(S:Ek) / S(rho_S)
```

At t=0, the system and environment are uncorrelated:
- rho_S,Ek = rho_S (x) rho_Ek (product state)
- I(S:Ek) = S(rho_S) + S(rho_Ek) - S(rho_S,Ek) = S(rho_S) + 0 - S(rho_S) - 0 = 0

Wait, this is wrong. Let me recalculate properly.

At t=0:
- rho_S = (|0><0| + |1><1| + |0><1| + |1><0|)/2 (pure state, S(rho_S) = 0)
- Actually no, S(|psi><psi|) = 0 for pure state

**Correction:** The system entropy S(rho_S) is:
- At t=0: rho_S is pure -> S(rho_S) = 0
- At t=t_dec: rho_S = (|0><0| + |1><1|)/2 -> S(rho_S) = 1 bit

This changes the interpretation. The normalized MI becomes:

```
E = I(S:Ek) / max(S(rho_S), epsilon)
```

At t=0: S(rho_S) = 0, so E is ill-defined (0/0).

**Resolution:** Use a different normalization. Let E be the absolute MI:

```
E_before = I(S:Ek) = 0  (no correlation)
E_after = I(S:Ek) = 1 bit (full correlation)
```

### 2.2 Handling grad_S

At t=0: All fragments have I(S:Ek) = 0
- grad_S_before = std([0, 0, ..., 0]) = 0
- With regularization: grad_S_before = epsilon

At t=t_dec: All fragments have I(S:Ek) ~ 1 bit
- grad_S_after = std([1, 1, ..., 1]) = 0
- With regularization: grad_S_after = epsilon

**Problem:** grad_S is the same before and after!

This is actually a feature: In Zurek's Quantum Darwinism, the DEFINING characteristic of classical reality is that ALL observers (fragments) agree. Low grad_S means consensus.

### 2.3 Computing the Ratio R_after / R_before

```
R = (E / grad_S) * sigma^Df
```

With the definitions above:

```
R_before = (0 / epsilon) * sigma^Df = 0

R_after = (1 / epsilon) * sigma^Df = sigma^Df / epsilon
```

Therefore:
```
R_after / R_before = infinity  (0 -> positive finite)
```

This is not a useful prediction because it depends on epsilon (the regularization constant).

---

## PART 3: A More Careful Derivation

The problem above is that E=0 at t=0 makes the ratio meaningless. Let me try a different approach that avoids this singularity.

### 3.1 Alternative Definition: E as Purity

**Definition:** E = Tr(rho_S^2), the purity of the system state.

- At t=0: rho_S is pure, Tr(rho_S^2) = 1
- At t=t_dec: rho_S = I/2, Tr(rho_S^2) = 1/2

But wait, this makes E DECREASE, which is opposite to "crystallization."

### 3.2 Alternative Definition: E as Distinguishability

**Definition:** E = 1 - S(rho_S)/log(d), where d=2 is Hilbert space dimension.

- At t=0: S(rho_S) = 0, so E = 1 (fully distinguishable)
- At t=t_dec: S(rho_S) = 1, so E = 0 (maximally mixed)

Again, E DECREASES. This doesn't capture crystallization.

### 3.3 The Correct Definition: E as Environmental Information

The key insight from Zurek: Classical reality emerges not in the SYSTEM but in the CORRELATIONS.

**Definition:** E = <I(S:Ek)>_k, the average mutual information across fragments.

- At t=0: <I(S:Ek)> = 0
- At t=t_dec: <I(S:Ek)> = S(rho_S) = 1 bit

Now E INCREASES, capturing crystallization.

### 3.4 The Full Prediction

With the definitions:
- E = <I(S:Ek)>_k (average MI)
- grad_S = std(I(S:Ek)) + epsilon (dispersion)
- Df = log(N+1)
- sigma = 1/2 (fixed)

**At t = 0:**
```
E_0 = 0
grad_S_0 = epsilon  (no variation when all are 0)
R_0 = (0 / epsilon) * (1/2)^{log(N+1)} = 0
```

**At t = t_dec:**
```
E_dec = 1 bit  (assuming S(rho_S) -> 1)
grad_S_dec = epsilon  (all fragments agree)
R_dec = (1 / epsilon) * (1/2)^{log(N+1)}
      = (1/epsilon) * (N+1)^{-log2}
      = (1/epsilon) * (N+1)^{-0.693}
```

**The Ratio:**
```
R_dec / R_0 = infinity (since R_0 = 0)
```

### 3.5 Making a Finite Prediction

To get a finite prediction, we need E_0 > 0. This happens if we consider the TRANSITION, not the extreme endpoints.

**At intermediate time t (0 < t < t_dec):**

During evolution, the fragments don't all gain information at exactly the same rate (due to different coupling constants or initial conditions).

Let's say at time t:
- I(S:E1) = a
- I(S:E2) = b
- ...
- I(S:EN) = some distribution

Then:
```
E(t) = mean([a, b, ...])
grad_S(t) = std([a, b, ...])
```

**Key Physical Insight:** During decoherence:
1. E(t) increases from 0 to 1
2. grad_S(t) first INCREASES (fragments desynchronize) then DECREASES (fragments synchronize)

The ratio E/grad_S depends on the relative rates.

### 3.6 A Testable Prediction

**PREDICTION:** The maximum of dR/dt occurs at time t* where:

```
d/dt [E(t) / grad_S(t)] is maximized
```

For Zurek's Quantum Darwinism with identical coupling:
- All fragments gain information at the same rate
- grad_S(t) stays small throughout
- R(t) ~ E(t) * constant
- dR/dt ~ dE/dt

Therefore:
```
t* = time of maximum dI/dt
```

For CNOT-like interaction H = g * sigma_z * sigma_x:
```
I(S:E) ~ sin^2(g*t)   (to leading order)
dI/dt ~ 2*g*sin(g*t)*cos(g*t) = g*sin(2*g*t)
Maximum at: 2*g*t = pi/2, so t* = pi/(4*g)
```

**Testable Prediction:**
```
The R-spike occurs at t* = pi / (4 * g * N)

where g is coupling strength, N is number of environment modes.
```

For the simulation parameters (g=0.5, N=6):
```
t* = pi / (4 * 0.5 * 6) = pi / 12 = 0.262
```

Compare to simulation's reported t_dec ~ 0.52 = pi/(2*g*N) = pi/6.

**Discrepancy:** The predicted t* = 0.262 is HALF of the simulated t_dec = 0.52.

This is because:
- t_dec is defined as coherence drop to 1/e
- t* is maximum rate of MI increase
- These are different quantities!

---

## PART 4: Comparison with Zurek's Quantum Darwinism

### 4.1 What Zurek Predicts

Zurek's Quantum Darwinism makes these predictions:

1. **Plateau in MI vs. fragment fraction:** I(S:F) saturates at S(rho_S) once fragment is large enough
2. **Redundancy:** R_delta = number of fragments that can independently determine system state
3. **Decoherence rate:** Gamma ~ g^2 * N (for weak coupling) or Gamma ~ g*N (for strong)
4. **Objectivity criterion:** Classical if R_delta >> 1

### 4.2 What the R Formula Predicts

The R formula R = (E/grad_S) * sigma^Df predicts:

1. **R increases during decoherence** (as E increases and grad_S stays low)
2. **R ~ (N+1)^{-log(sigma)}** scaling with environment size
3. **dR/dt peaks at t* = pi/(4gN)** (from derivation above)
4. **R correlates with Zurek redundancy** (by construction, since E = <MI>)

### 4.3 Where Do They Differ?

**CRITICAL ANALYSIS:**

| Quantity | Zurek Predicts | R Formula Predicts | Same? |
|----------|---------------|-------------------|-------|
| MI saturation | Yes, at S(rho_S) | E -> 1 | YES (by definition) |
| Timescale | t_dec ~ 1/(gN) | t* ~ 1/(gN) | YES (same scaling) |
| Fragment dependence | I(S:F) ~ S(rho_S) | E = <I(S:Ek)> | YES (average) |
| Redundancy | R_delta = N/ln(N) | sigma^Df ~ N^{-0.693} | DIFFERENT! |

**The Key Difference:**

Zurek's redundancy R_delta ~ N/ln(N) (grows with N)
R formula's sigma^Df ~ N^{-0.693} (DECREASES with N)

This is a testable distinction!

### 4.4 The Novel Prediction

**DERIVED PREDICTION (Not from simulation):**

```
If sigma < 1, then R DECREASES with increasing environment size N.
If sigma > 1, then R INCREASES with increasing environment size N.
```

For sigma = 0.5:
```
R ~ (N+1)^{-0.693}
```

This means:
- N=2: R ~ 0.62 * (E/grad_S)
- N=6: R ~ 0.36 * (E/grad_S)
- N=10: R ~ 0.28 * (E/grad_S)

**CONTRAST WITH ZUREK:**
- Zurek says MORE environment modes = MORE classical (higher redundancy)
- R formula (with sigma=0.5) says MORE modes = LOWER R

**Test:** Run simulation with varying N and check if R_final increases or decreases.

---

## PART 5: What the Existing Tests Actually Show

### 5.1 Audit of Test C

Looking at test_c_zurek_data.py:

```python
def compute_R_mi(state, n_total, sigma=0.5):
    # ...
    E_mi = np.mean(mi_array)
    grad_mi = np.std(mi_array) + 0.01
    Df = np.log(n_env + 1)
    return (E_mi / grad_mi) * (sigma ** Df)
```

This implements exactly what I derived above. The test shows R_mi increases 2.06x during decoherence.

**But this is NOT a test of the formula!**

The test shows: E_mi increases (which we know from Zurek)
The test shows: grad_mi stays low (which we know from QD)
The test shows: sigma^Df is constant (by definition)

So R_mi = (increasing) / (constant) * (constant) = increasing.

This is TAUTOLOGICAL. It confirms that <MI> increases during decoherence, which is Zurek's result, not a novel prediction.

### 5.2 What Would Be a Real Test

A real test would check:

1. **The sigma^Df factor:** Does R scale as (N+1)^{-0.693}?
   - Run with N=2, 4, 6, 8, 10
   - Plot R_final vs N
   - Check if slope matches prediction

2. **The t* prediction:** Does dR/dt peak at pi/(4gN)?
   - Extract t* from simulation
   - Compare to analytical prediction
   - Check if they match

3. **The Zurek contrast:** Does R behave opposite to Zurek redundancy for large N?
   - Zurek: redundancy increases with N
   - R formula: R decreases with N (if sigma<1)
   - Which matches experiment?

---

## PART 6: The Honest Conclusion

### 6.1 What We Have Derived

From the R formula R = (E/grad_S) * sigma^Df applied to decoherence:

1. **E = <I(S:Ek)>** increases from 0 to S(rho_S) during decoherence
2. **grad_S** stays small (consensus) in Quantum Darwinism
3. **sigma^Df = sigma^{log(N+1)} = (N+1)^{log(sigma)}** is a power law in N

### 6.2 Novel Predictions

**Prediction 1:** R scales as N^{log(sigma)} with environment size.
- For sigma=0.5: R ~ N^{-0.693}
- This CONTRADICTS Zurek if sigma < 1

**Prediction 2:** The R-spike occurs at t* = pi/(4gN), which is HALF the decoherence time.

**Prediction 3:** R_after/R_before is NOT universal; it depends on N via sigma^Df.

### 6.3 Falsification Criteria

The R formula (as applied to decoherence) would be FALSIFIED if:

1. R does NOT scale as N^{log(sigma)} with environment size
2. The R-spike does NOT occur at t* = pi/(4gN)
3. Larger environments show HIGHER R (instead of lower for sigma<1)

### 6.4 The Fundamental Issue

The R formula R = (E/grad_S) * sigma^Df is not TESTED by the current simulations because:

1. **E is defined as <MI>** which increases by Zurek's theorem
2. **grad_S is defined as std(MI)** which stays low by QD definition
3. **sigma and Df are free parameters** that can be tuned

The formula adds the factor sigma^Df on top of Zurek's theory. This factor is:
- Untested (no comparison to experiment)
- Unmotivated (why sigma = 0.5? why Df = log(N+1)?)
- Potentially contradictory to Zurek (R decreases with N if sigma<1)

### 6.5 What Must Be Done

To actually test the R formula:

1. **Fix sigma from first principles** (not a free parameter)
2. **Fix Df from system properties** (not log(N+1) by fiat)
3. **Derive a quantity that differs from Zurek**
4. **Test against experimental data** (not simulations)

Until then, the formula remains an **unfalsified hypothesis** dressed in the language of validated theory.

---

## PART 7: Summary Table

| Question | Answer |
|----------|--------|
| What is E operationally? | E = <I(S:Ek)>, average mutual information |
| What is grad_S operationally? | grad_S = std(I(S:Ek)), dispersion of MI |
| What is sigma operationally? | sigma = 0.5 (arbitrary parameter) |
| What is Df operationally? | Df = log(N+1) (arbitrary choice) |
| What is R before decoherence? | R_0 = 0 (since E=0) |
| What is R after decoherence? | R_dec = sigma^Df / epsilon |
| What is R_after/R_before? | Ill-defined (infinity) |
| What does the R formula predict? | R ~ N^{log(sigma)} scaling |
| What does Zurek predict? | Redundancy ~ N/ln(N) |
| Are they the same? | NO! Opposite N-dependence if sigma<1 |
| Is this tested? | NO. Current tests confirm Zurek, not R formula. |

---

## PART 8: The Path Forward

To make progress, we need:

### 8.1 Immediate Steps

1. **Run N-dependence test:** Vary N from 2 to 20, plot R_final vs N
2. **Check sigma^Df prediction:** Fit R ~ N^alpha, extract alpha, compare to log(sigma)
3. **Check t* prediction:** Extract time of max dR/dt, compare to pi/(4gN)

### 8.2 Theoretical Work

1. **Derive sigma from first principles:** Why should sigma = 0.5? What is sigma physically?
2. **Derive Df from system properties:** Why log(N+1)? What about different couplings?
3. **Address the Zurek contradiction:** If sigma<1, R decreases with N. Is this physical?

### 8.3 Experimental Comparison

1. **Find real QD data with varying N:** Photonic, atomic, or NV center experiments
2. **Extract R from experimental MI curves**
3. **Check if R scales as predicted**

---

*Created: 2026-01-30*
*Purpose: Honest derivation of testable predictions from R formula*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
