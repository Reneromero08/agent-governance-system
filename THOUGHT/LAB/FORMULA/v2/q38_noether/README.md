# Q38: R Obeys Noether Conservation Laws

## Hypothesis
The semiosphere (semantic embedding space) obeys Noether conservation laws. Specifically: concepts follow geodesic motion on the embedding sphere, the symmetry is rotational SO(d), and the conserved quantity is angular momentum magnitude |L| = |v|. A Lagrangian L = (1/2)|v|^2 governs semantic dynamics, and the action principle applies to meaning evolution.

## v1 Evidence Summary
- 6/6 synthetic tests passed: geodesics stay on sphere (deviation 10^-16), angular momentum conserved (CV = 2.5e-15), plane angular momentum conserved (10/10 planes), speed conserved (CV = 3.3e-15), non-geodesics violate conservation (CV = 0.04), geodesics minimize action.
- Cross-architecture validation on 5 models (GloVe, Word2Vec, FastText, BERT, SentenceTransformer): SLERP CV ~5e-7, mean separation 69,000x between geodesic and perturbed paths.
- Original hypothesis (scalar momentum conservation in flat principal subspace) was falsified: principal subspace deviation from flat = 1.45, scalar momentum CV = 0.83.

## v1 Methodology Problems
1. **Core claim is tautological.** "Angular momentum is conserved along geodesics on the unit sphere" is a theorem of differential geometry, true for any Riemannian manifold with L = (1/2)|v|^2. The CV = 10^-15 result confirms NumPy trigonometric functions work, not that embeddings have a physical property.
2. **Cross-architecture test is circular.** The test constructs SLERP interpolations (which are geodesics by definition) between embedding endpoints, then checks if those geodesics conserve angular momentum. This would pass with any two unit vectors, including random ones. The embeddings play no role in the result.
3. **No time evolution is defined.** Embeddings are static vectors. There is no Hamiltonian, no equation of motion, no observed temporal trajectory. SLERP is a mathematical interpolation chosen by the experimenter, not an observed physical trajectory.
4. **Lagrangian is asserted, not derived.** L = (1/2)|v|^2 is the simplest possible Lagrangian on a manifold. No justification is given for why semantic dynamics should follow this Lagrangian.
5. **SO(d) symmetry is broken in practice.** Embedding spaces are not rotationally symmetric -- different directions carry different semantic information. PCA eigenvalue spectra explicitly show highly non-uniform direction importance, breaking SO(d).
6. **Interpretive overclaims.** "Truth flows freely; lies fight the geometry," "lie detection via conservation violation," and "meaning has inertia" are metaphors presented as findings. No deceptive vs. truthful trajectories were tested.
7. **Loop tests failed.** Multi-segment analogy loops (king->queen->woman->man) failed conservation in all 5 architectures (CV 0.076-0.285), contradicting the claim that meaning follows geodesics.
8. **Perturbed trajectories also passed.** In cross-architecture tests, perturbed paths had CV ~0.03-0.05, which is below the stated threshold of 0.05. The negative control fails to discriminate.

## v2 Test Plan

### Test 1: Observed Trajectory Analysis
- Track actual temporal evolution of word embeddings across training epochs (e.g., using checkpoints from BERT/GPT training, or diachronic embeddings like HistWords).
- Measure whether observed trajectories are closer to geodesics than to random walks or straight-line interpolations in ambient space.
- Compute angular momentum CV along observed (not constructed) trajectories.

### Test 2: Geodesic vs. Alternative Path Comparison
- For word analogy tasks (king-queen-man-woman), compute the actual embedding trajectory vs. the geodesic prediction.
- Compare geodesic interpolation to linear interpolation, cubic spline, and random walk as alternative models of semantic motion.
- Use held-out analogy accuracy as the evaluation metric.

### Test 3: Symmetry Breaking Characterization
- Measure the actual symmetry group of the embedding space by analyzing eigenvalue spectra and direction importance.
- Quantify how far the space is from SO(d) invariance.
- Identify the largest subgroup that is approximately preserved and derive the corresponding Noether charge.

### Test 4: Conservation Along Real Contextual Shifts
- Extract BERT/GPT contextual embeddings for the same word across 1000+ different sentences.
- Treat contextual variation as a trajectory through embedding space.
- Measure whether any quantity (speed, angular momentum, energy) is approximately conserved along these contextual trajectories.
- Compare conservation quality to shuffled controls.

### Test 5: Lagrangian Derivation Test
- From observed trajectories (Test 1 or 4), use inverse methods to infer the Lagrangian.
- Test whether the inferred Lagrangian is L = (1/2)|v|^2 or something else.
- Compare against at least 3 alternative Lagrangians (e.g., L = |v|, L = (1/2)|v|^2 + V(x), L with direction-dependent mass tensor).

## Required Data
- **HistWords** diachronic word embeddings (1800-2000, decade snapshots) -- Hamilton et al.
- **BERT/GPT training checkpoints** -- either train a small model saving every N steps, or use publicly available training logs
- **STS Benchmark** -- semantic textual similarity for trajectory evaluation
- **Google Analogy Test Set** -- word analogy tasks for geodesic prediction accuracy
- **BATS** (Bigger Analogy Test Set) -- 99,200 analogy questions
- **Contextual embeddings** from BERT/GPT on standard corpora (WikiText-103, Penn Treebank)

## Pre-Registered Criteria
- **Success (geodesic motion):** Observed temporal trajectories have angular momentum CV < 0.10, AND geodesic predictions outperform linear interpolation on analogy tasks by > 5% accuracy.
- **Failure (geodesic motion):** Angular momentum CV > 0.30 on observed trajectories, OR geodesic predictions perform worse than linear interpolation.
- **Success (conservation):** At least one quantity conserved along contextual trajectories with CV < 0.15, significantly better than shuffled controls (p < 0.01).
- **Failure (conservation):** No quantity conserved better than shuffled controls at p < 0.05.
- **Inconclusive:** Marginal improvements over baselines (5-15% range) with large variance.

## Baseline Comparisons
- **Linear interpolation** in ambient space (the simplest possible "trajectory" model).
- **Random walk** on the unit sphere (null model for trajectory structure).
- **PCA-projected straight line** (accounts for dimension importance without geodesic structure).
- **Shuffled contextual embeddings** (destroys temporal structure, baseline for conservation).

## Salvageable from v1
- noether.py contains correct implementations of sphere geodesics, angular momentum computation, and action integrals -- useful as a computational library.
- test_q38_real_embeddings.py has working code for loading GloVe, Word2Vec, FastText, BERT, and SentenceTransformer embeddings -- reusable infrastructure.
- The falsification of scalar momentum conservation is a genuine negative result worth documenting.
- The observation that loop tests fail conservation is informative and should be investigated further.
