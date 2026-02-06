# Q45: Geometry Alone Suffices for Semantic Navigation

## Hypothesis

After initializing with embedding vectors, the semantic manifold can be navigated using only geometric operations (vector arithmetic, interpolation, dot products) without returning to the language encoder. Embeddings serve as "GPS coordinates" -- once initialized, all reasoning happens in pure geometry.

## v1 Evidence Summary

Tests across 5 embedding architectures (MiniLM-L6, MPNet-base, Paraphrase-MiniLM, MultiQA-MiniLM, BGE-small) reported 100% pass rate on 4 tests:

- **Composition (Test 1):** Vector arithmetic (king - man + woman -> queen). 4/4 test cases passed for all 5 models.
- **Superposition (Test 2):** Vector averaging ((cat + dog)/norm -> pet/animal). 4/4 passed for all 5 models.
- **Geodesic (Test 3):** Slerp interpolation midpoint (hot <-> cold -> warm/cool). 4/4 passed for all 5 models.
- **E-Gating (Test 4):** Cohen's d between related vs unrelated pair dot products ranged from 4.53 to 7.47 across models. All models passed d > 0.8 threshold.

The original Test 4 using the full R formula failed due to sigma^Df = 1.73^200 = 10^47 numerical explosion; the test was changed to use E (cosine similarity) alone.

## v1 Methodology Problems

The Phase 3 verdict identified severe issues:

1. **The central claim is a tautology:** Embedding models are DESIGNED so that cosine similarity tracks semantic similarity and vector arithmetic produces meaningful analogies. This has been known since word2vec (Mikolov et al., 2013). Demonstrating that these operations work is validating the design specification, not discovering a new property.
2. **Cherry-picked test cases with generous acceptance:** Expected word lists included input terms themselves (e.g., "woman" as valid output for "king - man + woman"). For geodesics, endpoints dominated the top results (positions 1-4) with the intermediate word barely appearing at position 5. The 100% pass rate reflects generous acceptance criteria, not strong signal.
3. **E-gating test is circular:** E is cosine similarity. Testing whether cosine similarity distinguishes related from unrelated pairs is testing the design objective of every embedding model. The large Cohen's d values (4.5-7.5) validate that sentence-transformers work as designed, not the Born rule.
4. **"Navigate" is never defined:** No path-planning, multi-step reasoning, obstacle avoidance, or traversal of non-obvious connections. The tests show that single-step vector operations produce nearby vectors -- not navigation in any meaningful sense.
5. **R formula not tested:** The full R formula (R = (E/grad_S) * sigma^Df) was abandoned due to numerical explosion. Only E (cosine similarity) was tested. R=1900 is assigned to a test that does not test R.
6. **All "quantum" vocabulary is metaphorical:** "Superposition" = vector averaging, "Born rule" = dot product, "geodesic" = slerp on a sphere. None are used in their technical physics sense.
7. **No difficult test cases:** All examples are "obvious" (cat/dog, king/queen, hot/cold). No adversarial cases, ambiguous terms, polysemous words, or boundary conditions.

## v2 Test Plan

### Experiment 1: Multi-Step Geometric Reasoning

Test whether geometric operations maintain coherence over multiple sequential steps (not just one-step analogies).

- **Method:** Define multi-step reasoning chains (e.g., start at "puppy", apply 3-5 sequential vector operations to navigate to "elderly"). Measure semantic coherence at each step and whether the final destination is reached.
- **Data:** 50 multi-step paths of varying length (3, 5, 7, 10 steps) across 3 models
- **Analysis:** Plot coherence (cosine similarity to intended path) vs number of steps. Compare to null model of random walks on the sphere. Measure at what step count geometric navigation degrades to random.
- **Key question:** How many sequential geometric operations can be chained before the result becomes meaningless?

### Experiment 2: Geometric Navigation vs Encoder Re-querying

Compare pure geometric navigation to the alternative of re-encoding at each step.

- **Method:** For each multi-step task, compute the result two ways: (a) purely geometric (vector operations only), (b) re-encode intermediate results as text and re-embed. Compare accuracy on downstream tasks.
- **Data:** Analogy completion, sentence similarity, simple QA tasks
- **Analysis:** If geometric navigation matches re-encoding accuracy, geometry suffices. If re-encoding is substantially better, geometric "GPS coordinates" are insufficient.
- **Key question:** Does the "GPS coordinates" metaphor hold, or does the encoder add information at each step?

### Experiment 3: Hard Test Cases

Test geometric operations on cases where they are known to struggle.

- **Method:** Test composition, interpolation, and navigation on: (a) polysemous words (bank = river bank vs financial bank), (b) abstract concepts (justice, freedom, entropy), (c) negation (happy -> not happy), (d) rare words, (e) out-of-distribution phrases, (f) long compositional phrases
- **Data:** 100 hard test cases per category across 5 models
- **Analysis:** Report pass rates with strict acceptance criteria (correct answer must be top-1, not top-5). Compare to easy cases from v1 to quantify the difficulty gradient.
- **Key question:** Where does geometric navigation break down?

### Experiment 4: Navigation with Quantified Drift

Measure how compositional error accumulates in geometric operations.

- **Method:** Compute "compositional drift" = cosine distance between the geometric result and the directly encoded result for increasingly complex operations (1 operation, 2 composed, 3 composed, ..., 10 composed).
- **Data:** 200 compositional chains across 3 models
- **Analysis:** Fit drift model (linear, exponential, etc.). Determine the practical limit of composition before drift exceeds a meaningful threshold (e.g., result closer to random than to target).

### Experiment 5: Comparison to Known Baselines

Compare geometric navigation to established NLP methods on standard benchmarks.

- **Method:** On analogy completion (Google analogies, BATS), compare: (a) vector arithmetic (3CosAdd, 3CosMul), (b) slerp interpolation, (c) lookup-based approaches, (d) LLM direct completion
- **Data:** Full Google analogy dataset (19,544 questions), BATS (99,200 questions)
- **Analysis:** Report accuracy with standard metrics. These benchmarks have published baselines from the literature.

## Required Data

- Google analogy dataset (19,544 word analogies)
- BATS analogy dataset (99,200 balanced analogies)
- STS Benchmark for similarity evaluation
- WordSim-353, SimLex-999 for similarity baselines
- 5+ embedding models: MiniLM, MPNet, BGE, GTE, E5

## Pre-Registered Criteria

- **Success (confirm):** Multi-step navigation maintains > 0.5 cosine similarity to intended path for >= 5 steps AND geometric navigation matches re-encoding accuracy within 10% on downstream tasks AND hard test cases achieve >= 40% top-1 accuracy (strict)
- **Failure (falsify):** Navigation degrades to random within 3 steps OR re-encoding outperforms geometry by > 25% OR hard test cases achieve < 20% top-1 accuracy
- **Inconclusive:** Navigation works for 3-4 steps but degrades beyond; re-encoding advantage is 10-25%; hard cases score 20-40%

## Baseline Comparisons

- **Random walk on sphere:** Random unit vectors added at each step (null model for multi-step navigation)
- **Encoder re-querying:** Re-encode intermediate text results at each step (gold standard comparison)
- **Published analogy baselines:** 3CosAdd, 3CosMul, LRCos from Mikolov et al. (2013) and subsequent work
- **LLM direct completion:** GPT-4 or Claude completing analogies directly (language-based upper bound)

## Salvageable from v1

- The multi-architecture test framework (5 models) is reusable
- The observation that E (cosine similarity) discriminates related from unrelated with large effect sizes is valid (it validates embedding model design)
- The slerp implementation for geodesic interpolation
- The finding that the full R formula (sigma^Df) is numerically unstable is an important negative result worth preserving
- Test script: `test_pure_geometry_multi_arch.py`
