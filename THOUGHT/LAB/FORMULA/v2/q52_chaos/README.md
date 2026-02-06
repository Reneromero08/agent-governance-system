# Q52: R Correlates with Chaos Measures

## Hypothesis

The participation ratio R correlates with chaos measures in dynamical systems. Originally hypothesized as an inverse correlation (R decreases with chaos/Lyapunov exponent), the v1 investigation falsified this and found a positive correlation instead. The refined hypothesis is: R (participation ratio) tracks the effective dimensionality of dynamical attractors and can estimate fractal dimensions of strange attractors.

## v1 Evidence Summary

Testing on standard dynamical systems benchmarks:

- **Logistic map sweep (r = 2.5 to 4.0, n=100 points):** Pearson r = +0.5449 (p = 4.6e-09), Spearman rho = +0.6294 (p = 2.3e-12). Strong POSITIVE correlation -- opposite of the pre-registered hypothesis (r < -0.5).
- **Bifurcation detection:** Only 1/4 known bifurcations detected (first period-doubling at r=3.0, |dR/dr| = 66.00 vs threshold 15.37). Second bifurcation, onset of chaos, and full chaos transitions all missed.
- **Henon attractor:** Chaotic (a=1.4): Lyapunov = +0.4147, R = 1.160. Regular (a=0.2): Lyapunov = -0.216, R = 0.000. Confirms positive direction.
- **Negative control (random noise):** Mean R = 2.9993, CV = 0.0002. Consistent and near maximum for 3D embedding.
- **Lyapunov accuracy:** At r=4.0, computed Lyapunov = 0.6932 vs theoretical ln(2) = 0.6931 (0.004% error). Computation is sound.
- **Henon R vs fractal dimension:** R = 1.16 vs known fractal dimension 1.26. Approximate but not exact correspondence.

## v1 Methodology Problems

The Phase 6C verdict confirmed the falsification but noted:

1. **Numerical clipping:** np.clip(x, 1e-10, 1-1e-10) subtly changes the dynamics of the logistic map at r=4. While scientifically justified for numerical stability, the test measures "clipped logistic map" not exact logistic map.
2. **Bifurcation detection overclaimed:** 1/4 bifurcations detected, but the section heading "Bifurcation Detection" could mislead readers into thinking R is a general bifurcation detector. It only detects the most dramatic 0D-to-1D transition.
3. **Post-hoc Lorenz reinterpretation:** The earlier Lorenz test (R^2 = -9.74) was retroactively reinterpreted as "likely reflects embedding issues" without re-running the test. This is speculation, not evidence.
4. **Two hypotheses untested:** H3 (Edge of Chaos) and H5 (Sensitive Dependence) were listed but never tested, yet the question is marked RESOLVED.
5. **R = Df overclaim:** The document claims "R (participation ratio) = Df (effective fractal dimension) for ergodic systems." The participation ratio estimates the number of significant covariance eigenvalues, which correlates with but is not identical to fractal dimension. For the logistic map at r=4, R=2.999 matches embedding dimension 3, not the fractal dimension 1.0 of the attractor.

## v2 Test Plan

### Experiment 1: Systematic R vs Fractal Dimension Comparison

Rigorously compare the participation ratio R to known fractal dimensions across multiple dynamical systems.

- **Systems:** Logistic map (various r), Henon map (various a), Lorenz attractor (various rho), Rossler attractor, double pendulum, standard map
- **Method:** For each system at each parameter value, compute: (a) R from delay-embedded time series covariance, (b) correlation dimension (Grassberger-Procaccia), (c) box-counting dimension, (d) Lyapunov dimension (Kaplan-Yorke formula from Lyapunov spectrum)
- **Data:** 50,000+ time steps per system, proper delay embedding (mutual information for delay, false nearest neighbors for dimension)
- **Analysis:** Plot R vs each fractal dimension estimate. Compute correlation and systematic bias. Determine whether R consistently over- or under-estimates dimension.
- **Key question:** How accurately does R estimate fractal dimension, and what are the systematic biases?

### Experiment 2: Embedding Parameter Sensitivity

Test how sensitive R is to the delay embedding parameters.

- **Method:** For the Lorenz and Henon attractors, vary: (a) delay tau (1 to 50 time steps), (b) embedding dimension m (2 to 10), (c) time series length N (100 to 100,000)
- **Analysis:** Plot R vs each parameter. Identify the convergence behavior: at what N does R stabilize? How sensitive is R to the choice of tau and m?
- **Key question:** Are there standard embedding parameters that make R a reliable dimension estimator?

### Experiment 3: Edge of Chaos Detection (Untested H3)

Test whether R marks the edge of chaos boundary.

- **Method:** For the logistic map, compute R at fine resolution (delta_r = 0.001) near the onset of chaos (r = 3.56 to 3.58). For cellular automata (Wolfram rules), compute R for rules classified as Class I-IV and test whether R distinguishes Class IV (edge of chaos).
- **Analysis:** Plot R vs r near the chaos onset. Test whether R shows a discontinuity, divergence, or other signature at the critical parameter value. For cellular automata, report R distributions per Wolfram class.
- **Key question:** Can R detect the edge of chaos, as originally hypothesized in H3?

### Experiment 4: Sensitive Dependence Detection (Untested H5)

Test whether R can detect sensitive dependence on initial conditions.

- **Method:** For each dynamical system, run two trajectories from nearby initial conditions (delta_x = 1e-8). Compute R for the difference trajectory (x1(t) - x2(t)). Compare this "divergence R" between chaotic and regular regimes.
- **Analysis:** In chaotic regimes, the difference trajectory should fill more dimensions (higher R). In regular regimes, it should remain confined (lower R). Test this systematically.

### Experiment 5: Lorenz Attractor Re-Test

Re-run the failed Lorenz test with proper methodology.

- **Method:** Integrate the Lorenz system (sigma=10, rho=28, beta=8/3) for 100,000+ time steps. Apply proper delay embedding. Compute R and compare to the known Kaplan-Yorke dimension (2.06).
- **Analysis:** Determine whether the original R^2 = -9.74 failure was due to embedding issues (too few points, wrong delay, wrong dimension) or a genuine failure of R on this system.

## Required Data

- Standard dynamical systems libraries (scipy.integrate for ODEs, custom maps)
- Known fractal dimension values from the literature for validation
- Wolfram cellular automata rule tables (256 elementary rules)
- Time series: 50,000-100,000 steps per system for convergence

## Pre-Registered Criteria

- **Success (confirm R tracks dimension):** Pearson r > 0.85 between R and Kaplan-Yorke dimension across >= 5 dynamical systems AND R for Lorenz attractor = 2.06 +/- 0.3 AND R distinguishes Wolfram Class IV from Class I-III (p < 0.01)
- **Failure (falsify):** Pearson r < 0.6 between R and fractal dimension OR R for Lorenz deviates from 2.06 by > 50% OR no edge-of-chaos signature detected
- **Inconclusive:** Correlation is moderate (0.6-0.85); Lorenz approximation is rough; cellular automata results are mixed

## Baseline Comparisons

- **Correlation dimension (Grassberger-Procaccia):** Standard fractal dimension estimator for time series
- **Box-counting dimension:** Alternative dimension estimator
- **Kaplan-Yorke dimension:** Computed from full Lyapunov spectrum (theoretical gold standard)
- **False nearest neighbors:** Standard embedding dimension estimator

## Salvageable from v1

- The positive correlation finding (R increases with chaos) is genuine and well-measured
- The Lyapunov exponent computation is validated against theory (0.004% error)
- The logistic map sweep infrastructure is reusable
- The Henon attractor test confirms the direction
- The insight that R measures effective attractor dimensionality (not predictability) is correct
- Test script: `test_q52_chaos.py`
- Results: `q52_chaos_results.json`
