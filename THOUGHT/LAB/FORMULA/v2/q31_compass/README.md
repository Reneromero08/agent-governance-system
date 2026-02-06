# Q31: R Enables Compass-Mode Navigation

## Hypothesis
The R formula primitives (E, sigma, Df) can be extended from a binary gate ("act or don't act") into a compass that answers "which way." Specifically: an action-conditioned resonance R(s,a) can rank candidate transitions using local information, and argmax_a R(s,a) yields reliable navigation / optimization direction across multiple task families. The proposed compass formula is:

```
Direction = argmax_a [J(s+a) * alignment_to_principal_axes(s+a)]
```

where J(x, anchors) = mean cosine similarity between x and its k nearest anchors, and principal_axes = top eigenvectors of the embedding covariance matrix (approximately 22 effective dimensions for trained models).

## v1 Evidence Summary
- Claimed CONFIRMED via Q43 (Quantum Geometric Tensor) validation.
- Q43 showed 96.1% subspace alignment between QGT eigenvectors and MDS eigenvectors, eigenvalue correlation = 1.000.
- J coupling: 0.09 for random embeddings, 0.39 for trained models.
- Effective dimensionality: participation ratio = 22.2/768 for trained models, 99.2/768 for random, 62.7/768 for untrained BERT.
- Contextual phase selection boost: 4.13x J coupling variance increase with explicit context in prompts.
- Key discovery: "J measures density, not semantic organization" (untrained BERT has higher J than trained).

## v1 Methodology Problems
1. **No implementation exists**: The question explicitly requires "a reproducible construction where argmax_a R(s,a) yields reliable navigation." Zero implementations of the compass formula were built. The only code is a simple nearest-neighbor search that ignores J coupling and principal axes entirely.
2. **"Confirmation" is tautological**: Q43's "confirmations" (96.1% alignment, eigenvalue correlation 1.000, principal axes = covariance eigenvectors) are guaranteed by the SVD theorem and the definition of PCA. These are mathematical tautologies, not empirical discoveries.
3. **Zero navigation tests**: The success criterion requires navigation "across multiple task families." Zero task families were tested. No navigation benchmark exists.
4. **No action-conditioned testing**: All measurements are on static embeddings, not on action transitions. The "compass" is never used to make a navigation decision.
5. **J coupling is scalar, not directional**: A direction field requires a vector-valued function. J is scalar. The gradient of J is never computed. The gradient of "mean cosine similarity to k nearest neighbors" is extremely noisy because which k neighbors are nearest changes discontinuously as x moves.
6. **PCA directions maximize variance, not goal proximity**: Following PCA directions is well-defined but does not minimize path length to any goal, which is what navigation requires.
7. **Formula mutates without control**: Three versions of the compass formula appear in the document, none tested.

## v2 Test Plan

### Test 1: Navigation Benchmark -- Word Analogy
- Task: navigate from word A to word B using only local compass readings.
- Start at embedding(A). At each step, compute compass scores for k candidate moves (nearest neighbors in vocabulary).
- Use compass formula: score(candidate) = J(candidate, anchors) * alignment_to_principal_axes(candidate).
- Measure: (a) number of steps to reach within cosine distance < 0.1 of target, (b) success rate across 1000 analogy pairs from Google Analogy dataset.
- Compare against baselines (see below).

### Test 2: Navigation Benchmark -- Semantic Path Following
- Task: given a sequence of waypoints (e.g., "dog" -> "animal" -> "biology" -> "science"), navigate through each waypoint in order.
- At each step, the compass should move toward the next waypoint using only local information (no direct access to the target embedding).
- Measure: total path length, waypoint hit rate, deviation from optimal geodesic path.
- Test on at least 3 task types: category ascent (dog -> mammal -> animal), analogy completion (king - man + woman -> queen), topic drift (politics -> economics -> finance).

### Test 3: Action-Conditioned R Comparison
- Define actions as movements in embedding space: a = {add context word, replace word, negate, specialize, generalize}.
- For each action at state s, compute R(s, a) using the proposed formula.
- Measure whether argmax_a R(s,a) selects the action that moves closest to a known target.
- Test on 500 (state, target) pairs sampled from STS Benchmark.

### Test 4: J Gradient Computation and Stability
- Implement the gradient of J with respect to position x in embedding space.
- Test gradient stability: compute nabla-J at 1000 random points. Report the fraction of points where the gradient direction flips (> 90 degrees) under small perturbation (1% of embedding norm).
- Compare J gradient with PCA principal axis alignment.
- Report whether the combined compass signal (J * PCA alignment) is more stable than either alone.

### Test 5: Cross-Task Navigation Performance
- Test the compass on at least 4 genuinely different task families:
  (a) Word analogy navigation (Google Analogy dataset)
  (b) Sentence similarity navigation (STS Benchmark)
  (c) Document topic navigation (20 Newsgroups)
  (d) Multilingual concept navigation (OPUS parallel corpus, navigate between translation equivalents)
- Report compass success rate and path efficiency for each task family.
- Determine whether a single compass parameterization works across tasks.

### Test 6: Contextual Phase Selection Validation
- Test whether adding explicit relational context (e.g., "in terms of biology") improves compass accuracy.
- Compare compass performance with and without context for each task family.
- Measure the 4.13x J coupling variance boost claim on held-out data.

## Required Data
- **Google Analogy Dataset** (~19K analogy quadruples, word2vec distribution)
- **STS Benchmark** (~8K sentence pairs, HuggingFace)
- **20 Newsgroups** (~18K documents, sklearn.datasets)
- **OPUS parallel corpus** (multilingual aligned sentences, open access)
- **GloVe pre-trained embeddings** (6B tokens, 300d, Stanford NLP)
- **all-MiniLM-L6-v2** and **all-mpnet-base-v2** (sentence-transformers)

## Pre-Registered Criteria
- **Success (confirm):** Compass navigation reaches the target (within cosine distance < 0.1) in at least 60% of test cases across at least 3 task families, AND outperforms the greedy baseline (direct cosine similarity to target) by at least 10% on path efficiency or success rate in at least 1 task family.
- **Failure (falsify):** Compass success rate < 40% on any task family, OR compass performs worse than or equal to greedy baseline on all task families, OR J gradient is unstable (flips > 50% of the time under 1% perturbation).
- **Inconclusive:** Compass works on word analogies but not on other task families, or compass matches but does not exceed greedy baseline.

## Baseline Comparisons
- **Greedy cosine similarity**: At each step, move to the neighbor with highest cosine similarity to the target (requires knowing the target embedding -- this is the "oracle" baseline).
- **Random walk**: Move to a random neighbor at each step.
- **PCA-only navigation**: Move along the principal axis direction closest to the target.
- **J-only navigation**: Move to the neighbor with highest J coupling (density-based).
- **Gradient descent on cosine distance**: Standard optimization baseline.

## Salvageable from v1
- The J coupling concept (mean cosine similarity to k nearest anchors) is well-defined and worth testing, even though it was never actually tested for navigation.
- The effective dimensionality measurements (Df = 22.2 for trained, 99.2 for random, 62.7 for untrained) are genuine empirical observations about embedding structure.
- The contextual phase selection finding (context in prompts changes J coupling variance) is testable.
- The code from Q43 for computing covariance eigenvectors and participation ratios is reusable.
