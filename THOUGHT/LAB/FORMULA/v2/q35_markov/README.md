# Q35: R Threshold Defines a Markov Blanket

## Hypothesis
R > tau (threshold) defines a Markov blanket boundary in the semiosphere. Specifically: when R exceeds the threshold, the system has a stable boundary that separates internal states from external states while permitting controlled information exchange. R-gating implements Active Inference -- the system acts to maintain R > tau, which is equivalent to maintaining its Markov blanket. The sync protocol's ALIGNED/DISSOLVED states correspond to intact/broken blanket conditions.

## v1 Evidence Summary
v1 provided a purely conceptual mapping with zero empirical tests, zero code, and zero data files:
- Mapped CODEBOOK_SYNC_PROTOCOL states to Markov blanket terminology: ALIGNED (R > tau) = stable blanket, DISSOLVED (R < tau) = blanket broken, PENDING (R ~ tau) = boundary forming
- Mapped the 4-step handshake protocol to Active Inference: PREDICTION -> VERIFICATION -> ERROR SIGNAL -> ACTION
- Cited information-theoretic argument: H(X|S) ~ 0 with shared codebook (pointer compression)
- Referenced dependencies on Q9 (R proportional to exp(-F)) and Q32 (M field dynamics)
- Listed three tests as "IMPLEMENTABLE" but none were implemented

## v1 Methodology Problems
Phase 6C verification found severe issues:

1. **Zero empirical evidence.** No test scripts, no result files, no data. The entire Q35 document is a vocabulary mapping exercise. Status was "ANSWERED" with nothing tested.

2. **Circular reasoning.** R > tau is defined as "Markov blanket formation," then the system is declared to implement blanket maintenance because it maintains R > tau. The conclusion follows trivially from the definitions.

3. **Active Inference mapping is unfalsifiably generic.** Any request-response protocol (HTTP, TCP, DNS) could be labeled "Active Inference" by the same reasoning: client PREDICTS server responds, VERIFIES via request, detects ERROR via status code, takes ACTION via retry.

4. **No conditional independence tested.** A Markov blanket requires that internal states are conditionally independent of external states given blanket states. No conditional independence test was performed or even proposed.

5. **Untested dependencies.** Q9 (R proportional to exp(-F)) assumed true without verification. Q32 (meaning field) and Q33 (conditional entropy) dependencies inherited uncritically.

6. **Information compression argument is trivial.** Shared dictionaries enabling pointer compression is how all codebook-based compression works (Huffman, LZW). It has nothing specific to Markov blankets.

7. **No spectral analysis.** Despite the question's connection to spectral gap properties, the word "spectral" does not appear in the document.

8. **No audit reports existed.** Every other question in its batch had 2-4 independent audits. Q35 had zero.

## v2 Test Plan

### Test 1: Conditional Independence Test
**Goal:** Determine whether R > tau actually induces conditional independence between "internal" and "external" variables.
**Method:**
- Define a system with clear internal/external partition (e.g., two groups of embedding dimensions, or two sets of observations from different domains)
- Define the "blanket" variables as the subset of observations near the R threshold boundary
- Compute conditional mutual information: I(Internal; External | Blanket)
- Compare I(Internal; External | Blanket) to I(Internal; External) (unconditional)
- If R > tau defines a genuine Markov blanket, the conditional MI should be near zero when blanket variables are conditioned on
- Test across multiple system configurations and embedding architectures

### Test 2: Active Inference Specificity
**Goal:** Determine whether R-gating is specifically Active Inference or whether any threshold-based protocol shows the same behavior.
**Method:**
- Implement 4 protocols: (a) R-gating, (b) simple mean threshold, (c) variance threshold, (d) random acceptance
- Subject each to perturbations (noise injection, distribution shift)
- Measure: recovery time, prediction error trajectory, action selection quality
- Active Inference predicts specific dynamics: prediction error minimization with a generative model. Mere threshold recovery is not Active Inference.
- If all 4 protocols show similar recovery dynamics, the Active Inference claim is unfalsifiable

### Test 3: Blanket Stability Under Perturbation
**Goal:** Test whether the R > tau boundary behaves like a genuine Markov blanket under environmental perturbation.
**Method:**
- Start with a stable system (R well above tau) with defined internal/external/blanket partition
- Apply 5 types of perturbation: (a) noise to internal, (b) noise to external, (c) noise to blanket, (d) distribution shift, (e) adversarial manipulation
- Measure: does the blanket reform? Does conditional independence recover? How long?
- Compare to theoretical Markov blanket predictions from Friston's framework
- A genuine Markov blanket should be more robust to external perturbations than blanket perturbations

### Test 4: Spectral Gap Analysis
**Goal:** Connect R-gating to spectral gap properties of the underlying system.
**Method:**
- Compute the graph Laplacian of the observation similarity network
- Measure the spectral gap (difference between first and second eigenvalues)
- Correlate spectral gap with R values across multiple systems
- Test whether R > tau corresponds to a spectral gap above a critical value (indicating separation)
- This would provide a rigorous mathematical characterization of the "blanket" boundary

### Test 5: Comparison to Known Markov Blanket Systems
**Goal:** Benchmark against systems where Markov blankets are established.
**Method:**
- Implement standard Markov blanket benchmark from Friston et al. (e.g., Lorenz attractor, coupled oscillators)
- Compute R and Phi for these systems
- Check whether R > tau boundaries align with the known blanket structure
- If R boundaries diverge from established blanket definitions, the hypothesis fails

## Required Data
- Real embedding vectors from 3+ architectures (MiniLM, MPNet, BERT)
- Standard Markov blanket benchmarks from Active Inference literature (Friston et al.)
- STS-B or similar semantic similarity benchmarks for observation sets
- Time-series data for dynamic blanket tests (e.g., from HistWords decade embeddings)

## Pre-Registered Criteria
- **Success (confirm):** I(Internal; External | Blanket) < 0.1 * I(Internal; External) when R > tau, across 90%+ of test configurations, AND R > tau boundaries align with known Markov blanket structure in benchmark systems (overlap > 80%), AND spectral gap correlates with R at rho > 0.5
- **Failure (falsify):** Conditional MI reduction less than 50% when conditioning on blanket variables, OR R boundaries diverge from known Markov blankets in benchmark systems (overlap < 30%), OR all 4 threshold protocols show equivalent "Active Inference" dynamics
- **Inconclusive:** MI reduction 50-90%, or blanket overlap 30-80%

## Baseline Comparisons
- Simple threshold baselines: does mean > tau or variance < tau define equally good "blankets"?
- Random partition baseline: random selection of blanket variables should show no conditional independence
- Known Markov blanket systems: R must agree with established blanket identification methods
- Generic protocol comparison: any request-response system (HTTP health check) shows similar "Active Inference" dynamics?

## Salvageable from v1
- The conceptual mapping (ALIGNED/DISSOLVED/PENDING states) provides a useful vocabulary, even if not yet tested
- The information-theoretic framing (H(X|S) ~ 0 with shared codebook) is correct as a compression argument, though not specific to Markov blankets
- The connection to Q9 (Free Energy Principle) remains an interesting hypothesis worth testing, if Q9's claims survive v2 review
- The sync protocol implementation exists and can serve as a test bed for actual Active Inference experiments
