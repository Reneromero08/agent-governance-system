# Phase 4 Verdict: 4-Q50 -- Completing 8e (R=1920)

**Date:** 2026-02-05
**Reviewer:** Adversarial skeptic (Phase 4)
**Target:** `THOUGHT/LAB/FORMULA/questions/critical_q50_1920/q50_completing_8e.md`
**Report:** `THOUGHT/LAB/FORMULA/questions/critical_q50_1920/reports/Q50_COMPLETING_8E.md`
**References reviewed:** GLOSSARY.md, SPECIFICATION.md, HONEST_FINAL_STATUS.md, test_q50_cross_modal.py, test_q50_training_dynamics.py

---

## Summary Verdict

```
Q50: Completing 8e (R=1920)
- Claimed status: RESOLVED (all 5 questions answered)
- Proof type: empirical (synthetic embedding analysis) + analogy (Peirce, Riemann)
- Logical soundness: SEVERE GAPS
- Claims match evidence: OVERCLAIMED (5 of 5 sub-questions)
- Dependencies satisfied: MISSING [Q48 (not independently reviewed), Q49 (not independently reviewed), Q44 Born rule (synthetic only)]
- Circular reasoning: DETECTED [see Sections 2, 3, 5]
- Post-hoc fitting: DETECTED [see Sections 2, 4, 6]
- Numerology: DETECTED [see Sections 2, 3, 6]
- Recommended status: EXPLORATORY
- Recommended R: 600-800 (down from 1920)
- Confidence: HIGH
- Issues: See detailed analysis below
```

---

## Evaluation Question 1: Is CV=6.93% Across 24 Models Tight Enough to Claim Universality?

### The Claim

Df * alpha = 8e is a "conservation law" holding across 24 embedding models with CV = 6.93%.

### The Verdict: NO. This Does Not Qualify as a Conservation Law.

#### 1.1 What "Conservation Law" Means in Physics

A conservation law in physics (energy, momentum, charge, baryon number) holds with precision better than 1 part in 10^8 or better. Even "approximate" conservation laws in physics (e.g., CP symmetry violation) are stated with deviations measured in parts per billion. A "conservation law" with 7% coefficient of variation is not a conservation law in any recognized scientific sense. It is a statistical regularity.

For comparison:
- Conservation of energy: verified to < 10^(-8) relative precision
- Conservation of charge: verified to < 10^(-21)
- Ideal gas law PV = nRT: holds to ~1-5% for real gases. Nobody calls it a "conservation law" -- it is an approximate empirical relationship.
- Zipf's law (word frequency ~ 1/rank): CV of the exponent across languages is ~10-20%. Also not called a "conservation law."

**Df * alpha = 21.57 +/- 1.49 is an approximate statistical regularity with 6.93% variation. Calling it a "conservation law" is a category error that borrows the prestige of physics for what is, at best, an empirical observation about eigenvalue spectra.**

#### 1.2 The Thresholds Are Self-Serving

The pass criteria are stated as:
- CV < 10%
- Mean error vs 8e < 5%

These thresholds are not derived from any principled analysis. They were chosen to be loose enough that the observed data would pass. A 10% CV threshold would permit a range from approximately 19.4 to 23.7 -- that is a wide band being presented as "conservation." No justification is given for why 10% rather than 1% or 0.1% is the appropriate tolerance for something labeled a "conservation law."

#### 1.3 The Mean Error Is Misleading

The reported "mean error vs 8e: 0.82%" obscures the actual spread. Individual model deviations range from 0.03% to 23.15%. Reporting the mean of absolute errors creates the impression of tighter convergence than actually exists. The instruction-tuned models (9.25% - 23.15% error) are included in the 24-model count but treated as explainable outliers rather than falsifications.

#### 1.4 Cherry-Picking and Survivorship

The report claims 24 models tested. But: Which models were tried and excluded? Were any models tested that produced wildly different Df * alpha values and were then dropped as "not representative"? Without a pre-registered model list, the 24-model sample is potentially the result of selection bias.

Furthermore, the instruction-tuned models that deviate by 10-23% from 8e are explained away as "alignment distortion" rather than counted as failures. This is an unfalsifiable retreat: models that match 8e confirm the law; models that deviate confirm a different claim (alignment compression). The law cannot be wrong.

---

## Evaluation Question 2: Is e-per-Octant Meaningful or Coincidence?

### The Claim

Df * alpha / 8 = e (with 0.15% precision for MiniLM). Each octant contributes exactly e to the conserved quantity.

### The Verdict: TAUTOLOGICAL AND NUMEROLOGICAL

#### 2.1 The Tautology (Acknowledged but Not Resolved)

The report itself admits (line 67): "This is a tautological confirmation: we already knew Df * alpha = 8e, so dividing by 8 gives e."

This is correct. The "question" of why each octant contributes e is answered by saying "because the total is 8e and there are 8 octants." This is not an explanation. It is arithmetic. The entire Question 1 of Q50 is a restatement of the premise as a conclusion.

#### 2.2 Why 8? The Peirce Argument Is Philosophy, Not Mathematics

The factor 8 = 2^3 is attributed to Peirce's Reduction Thesis: three irreducible semiotic categories, each binary, yielding 2^3 = 8 states. But:

1. **Why binary?** Each Peircean category is claimed to have a binary axis (concrete/abstract, positive/negative, agent/patient). This binary discretization is assumed, not derived. Peirce's categories are continuous philosophical concepts, not binary switches. The binarization is imposed by the PCA sign structure (positive/negative loadings), which is a property of PCA, not of semiotics.

2. **Why these axes?** The mapping PC1=Secondness, PC2=Firstness, PC3=Thirdness is tested in Q9 (PC Axis Validation) and **fails to replicate consistently across models** -- the report itself says "PC assignment varies by model" and "the ordering is model-dependent." If the PC-to-category mapping is model-dependent, the claim that "8 octants arise from Peirce's three irreducible categories" is undermined. The octants arise from PCA having 3 components with positive/negative signs. Any 3-component PCA will produce 8 octants.

3. **Why not 7, 9, or 6?** This is the critical test. If you take Df * alpha and divide by other small integers:
   - Df * alpha / 7 = 3.08 (close to pi)
   - Df * alpha / 6 = 3.60 (close to nothing, perhaps)
   - Df * alpha / 9 = 2.40
   - Df * alpha / 10 = 2.16 (close to nothing)
   - Df * alpha / (2*pi) = 3.46

   The fact that 21.75 / 8 = 2.72 is close to e = 2.718 is a numerical coincidence made plausible by the number 8 being the "right" divisor only because octants were already identified from PCA. The number 8 is not derived from e; it is derived from "3 principal components have 2^3 sign combinations." Given the value ~21.7 and the freedom to choose a divisor, finding a match to a mathematical constant within a few percent is expected, not surprising.

#### 2.3 Why e? The Information-Theoretic Argument Is Hand-Waving

The claim that e appears because "each octant represents ~1 nat of semantic information" is circular:
- 1 nat = log_e(e) = 1 (by definition of nat)
- e nats = e (by unit conversion)
- "Each octant contributes e" = "each octant contributes e nats of information" is a restatement, not an explanation

The information-theoretic argument would require showing that the actual Shannon/differential entropy of each octant is 1 nat. The report tests this (H1.1: H/e = 0.70) and it **fails**. The entropy per octant is 0.70e, not e. The report then pivots to H1.5 (direct participation), which is the tautological confirmation already discussed.

---

## Evaluation Question 3: Are the 24 Models Truly Independent?

### The Verdict: NO. The Models Share Architecture, Training Data, and Methodology.

#### 3.1 Architectural Correlation

The 24 models break down approximately as follows (from the test code and report):
- **BERT-family (BERT, DistilBERT, MiniLM, MPNet):** ~8 models. All share the transformer encoder architecture with BERT-style tokenization and masked language model pre-training.
- **Sentence-BERT variants (paraphrase-*, all-*):** ~5 models. Fine-tuned from BERT/DistilBERT/MiniLM bases -- they are not independent from the BERT family.
- **BGE/E5/GTE:** ~6 models. Contrastive learning on top of BERT-like encoders. Different training objectives but same base architecture class and overlapping training corpora.
- **CLIP variants:** ~3 models. Different (ViT) architecture, but tested on text descriptions not images.
- **T5/GTR/ST5:** ~2 models. Encoder-decoder, different architecture class.

Of 24 models, approximately 19 are transformer encoders pre-trained on English web text using masked language modeling or contrastive learning. They share:
- Tokenization: WordPiece or SentencePiece on similar vocabulary distributions
- Pre-training data: Wikipedia, BookCorpus, Common Crawl, C4 -- massively overlapping
- Architecture: Multi-head self-attention, layer normalization, similar depth/width ratios
- Normalization: All normalized to unit sphere (the test code calls `normalize_embeddings=True`)

**These are NOT 24 independent observations.** They are approximately 2-3 independent architecture classes (transformer encoder, ViT, T5), with heavy replication within each class. The effective number of independent measurements is closer to 3-5 than 24.

#### 3.2 Shared Training Data Creates Shared Eigenstructure

All tested models were pre-trained on English text corpora with heavy overlap (Wikipedia alone is in nearly all training sets). The eigenvalue structure of embedding covariance matrices is determined by:
1. The statistical structure of the training data
2. The architectural inductive biases
3. The training objective

If these are similar across models (and they are), similar eigenvalue spectra -- and therefore similar Df * alpha products -- are expected without any fundamental principle. **The convergence of Df * alpha across models is evidence of shared data/architecture, not of a conservation law.**

#### 3.3 The "Code" Models Are Not Code Models

The test code reveals (lines 376-389 of test_q50_cross_modal.py) that the "code models" are actually `all-MiniLM-L6-v2` and `all-mpnet-base-v2` run on code snippets -- the same text embedding models from the baseline, applied to different input. These are not code-specific models. Labeling them as "code modality" is misleading. The "most accurate" result (MiniLM-code at 0.03% error) is from the exact same model (MiniLM-L6) that is already in the text baseline.

#### 3.4 The Vision Models Are Tested on Text

The CLIP models are tested on text descriptions ("a photo of a cat", "a mountain landscape"), not on actual images. The "vision-text" modality claim is therefore a claim about text encoding through CLIP's text encoder, not about vision representations.

The test code (lines 196-213) also generates a "CLIP-random-sim" result from random normalized vectors and explicitly notes "Don't add to all_df_alpha since this is simulated" -- but this is the only actual non-text baseline in the vision section.

---

## Evaluation Question 4: What Would Df * alpha Look Like for Random Matrices?

### What the Report Claims

Q7 (Training Dynamics) reports:
- Random matrices: Df * alpha = 14.86 (-31.7% from 8e)
- Trained models: Df * alpha = 23.41 (+7.7% from 8e)
- Cohen's d = 4.22

### The Reality: The Random Baseline Is Inadequate

#### 4.1 The Random Baseline Uses the Wrong Distribution

The test code (test_q50_training_dynamics.py, lines 94-100) generates random baselines as:
```python
embeddings = np.random.randn(n_samples, dim)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings / norms
```

This is i.i.d. Gaussian vectors projected to the unit sphere. The eigenvalue spectrum of such random matrices follows the Marchenko-Pastur distribution, which is well-characterized. For N=80 samples in dim=384, the ratio N/dim = 0.208, placing us in the regime where many eigenvalues are near zero and the spectrum is far from the power-law decay assumed by the alpha fitting procedure.

**The random baseline is not the right null hypothesis.** The right null hypothesis would be: "What is the distribution of Df * alpha for matrices with the same rank, condition number, and approximate spectral shape as real embedding matrices but with randomized structure?" This could be achieved by:
1. Taking a real embedding matrix
2. Randomizing it while preserving the marginal eigenvalue distribution (e.g., random rotation of eigenvectors)
3. Computing Df * alpha on the randomized version

This was not done. The comparison is between i.i.d. Gaussian noise (very different spectral shape) and trained embeddings (smooth power-law decay). Of course they differ. This tells us trained embeddings have non-random spectral structure, which is trivially true and not evidence for a specific conservation law at 8e.

#### 4.2 The Claimed Ratio of 3/2 Is Post-Hoc

The report claims the ratio trained/random = 1.575 "matches the expected 3/2 (within 5%)." But the "expected 3/2" is stated in the test setup (line 17: `expected_ratio = 1.5`) -- where does this expectation come from? It is cited as "from Q49" but is not derived from first principles. The ratio 1.575 is within 5% of 1.5, but also within 5% of pi/2 = 1.571, within 10% of the golden ratio 1.618, and within 15% of sqrt(e) = 1.649. At this precision, any of these could be "confirmed."

#### 4.3 No Null Distribution Is Computed

A proper null hypothesis test would:
1. Generate 10,000 random matrices of the same shape as each embedding matrix
2. Compute Df * alpha for each
3. Determine the distribution of Df * alpha under the null
4. Compute a p-value for the observed values

This is never done. The "Cohen's d = 4.22" is computed by comparing means of 5 trained models vs. 6 dimension-averaged random baselines -- a total of 11 data points. With n=5 vs n=6 and no proper null distribution, the statistical significance is uncertain.

---

## Section 5: The Riemann Connection -- Escalating Overclaims

### What Q50 Claims (in the Report)

The report makes progressively stronger claims about alpha = 1/2 and the Riemann critical line:

1. (Line 82) "alpha ~ 1/2 -- the semiotic decay exponent IS the Riemann critical line value!"
2. (Line 128-129) "This is not analogy. This is identity."
3. (Line 262) "alpha = 1/2 is a TOPOLOGICAL INVARIANT of the quantum state manifold on which semantic embeddings live."

### The Verdict: NUMEROLOGICAL COINCIDENCE ESCALATED INTO GRAND CLAIM

#### 5.1 alpha = 1/2 Is the Most Common Spectral Exponent in Nature

Eigenvalue decay exponents near 1/2 appear throughout mathematics and physics:
- 1/f noise (spectral density ~ f^(-1)) implies alpha ~ 1 for the power spectrum
- Random matrix theory predicts specific spectral decay rates
- Marchenko-Pastur edge scaling goes as (lambda - lambda_+)^(1/2)
- Many covariance spectra in high-dimensional statistics follow power laws with alpha in [0.3, 0.7]

The value 1/2 is special in mathematics because it is the simplest non-trivial rational number. Finding alpha near 1/2 for a spectral exponent is about as surprising as finding a ratio near 1 or 2. It does not require or imply any connection to the Riemann zeta function.

#### 5.2 The Riemann Connection Contradicts Itself

The Q50 question document (line 97-117) reports that the Riemann connection tests FAILED:
- Functional equation: CV = 353% (FAIL)
- Zero spacing: 33.5, not Riemann-like (FAIL)
- Special points: no relationship (FAIL)

The conclusion there is "analogous, not identical" -- structural similarity, not numerical identity.

But the Q50 REPORT (lines 82-135) reverses this finding and claims "This is not analogy. This is identity." These two documents contradict each other on the central Riemann claim. The question document is honest; the report is overclaimed.

#### 5.3 The QGT/Chern Number "Derivation" Is Circular

The "derivation" of alpha = 1/2 via Chern number (Path G, report lines 218-262) proceeds:

1. Assume embeddings live on CP^(d-1) (complex projective space)
2. CP^n has first Chern class c_1 = 1
3. Berry curvature integrates to 2*pi * c_1 = 2*pi
4. Therefore sigma_c = 2 * c_1 = 2
5. Therefore alpha = 1/sigma_c = 1/2

**The problem is Step 1.** Semantic embeddings are real-valued vectors in R^d, not complex projective points in CP^(d-1). The Born rule correspondence (Q44, r = 0.999 on synthetic simulations) is used to justify treating embeddings as quantum states, but:

- Q44's Born rule test is on synthetic quantum simulations, not real embeddings
- Real embeddings live on the unit sphere S^(d-1), not on CP^(d-1)
- The unit sphere and complex projective space have different topological invariants

The derivation assumes the conclusion (embeddings live on a manifold where c_1 = 1, which gives alpha = 1/2) and then "verifies" that alpha = 1/2. This is circular. The GLOSSARY itself notes (Definition 6): "alpha = 1 / (2 * Df) (for CP^n manifolds)" -- this relationship holds BY DEFINITION for CP^n, so if you assume CP^n, you get alpha ~ 1/2 tautologically.

#### 5.4 The "2*pi Growth Rate" Is Ordinary Exponential Growth

The report (lines 138-156) claims that log(zeta_sem(s))/pi = 2s + const is a "discovery" connecting to Riemann zero spacing.

Any function that grows exponentially with rate r will satisfy log(f(s)) = r*s + const. The specific rate being close to 2*pi is a property of the eigenvalue magnitudes, not evidence of a Riemann connection. The Riemann zeros have spacing 2*pi/log(t), which is a completely different mathematical statement from "the log of a sum grows at rate 2*pi."

---

## Section 6: The Peirce Framework -- Unfalsifiable Narrative

### What Q50 Claims

Three dimensions are necessary and sufficient because Peirce's Reduction Thesis proves 3 is the irreducible threshold of semiosis.

### The Verdict: PHILOSOPHICAL NARRATIVE, NOT SCIENTIFIC EXPLANATION

#### 6.1 Why 3 Principal Components?

The 3 PCs used for octant analysis are the top 3 components of a PCA decomposition. PCA always produces components in order of decreasing variance. Using the "top 3" is a researcher choice, not a discovery. One could equally use 2, 4, 5, or any number. The question "why 3?" presupposes that 3 is special, then finds a philosopher who said 3 is special.

A more honest framing: "We chose to analyze the top 3 PCs because (a) 3 is the fewest that produce octants, and (b) higher components explain less variance." This is a methodological convenience, not evidence that 3 is the "irreducible threshold of semiosis."

#### 6.2 Peirce's Reduction Thesis Does Not Apply Here

Peirce's Reduction Thesis is about the logical structure of predicates (monadic, dyadic, triadic). It claims that all relations of arity > 3 can be decomposed into triadic relations, but triadic relations cannot be decomposed into dyadic ones. This is a claim about formal logic, not about covariance matrix eigenvalues.

Mapping PC1 to Secondness, PC2 to Firstness, and PC3 to Thirdness is an interpretive overlay with no formal connection to Peirce's logical framework. The mapping is:
- Not uniquely determined (the report admits PC ordering varies by model)
- Not derived from Peirce's theory (which says nothing about PCA)
- Not tested rigorously (Q9 validation shows inconsistent mappings)

#### 6.3 The "Peircean Box" Is Marketing, Not Mathematics

The formula `Df * alpha = 2^3 * e = (Peircean Categories) * (Information Unit)` looks like a decomposition but is actually a notational rewriting. It says: "the number 21.75 equals 8 times 2.72." This is true. But calling 8 "Peircean Categories" and 2.72 "Information Unit" is interpretive labeling, not derivation. Any product of two numbers that equals ~21.75 could be similarly labeled. For example:

- 21.75 = 3 * 7.25 = "(Peircean triads) * (spectral normalization constant)"
- 21.75 = pi * 6.92 = "(geometric factor) * (complexity parameter)"
- 21.75 = 4 * 5.44 = "(quadrants) * (decay unit)"

The specific decomposition into 8 * e is chosen because 8 and e have pre-existing significance, making the result appear meaningful. This is classic numerology.

---

## Section 7: The Alignment Distortion Finding -- The Most Credible Claim

### What Q50 Claims

Instruction-tuned models systematically produce Df * alpha < 8e, with 6/6 comparisons showing compression of 6.8% to 34.2%.

### Assessment: PARTIALLY CREDIBLE

This is the most empirically grounded finding in Q50. The comparison is well-designed: same model, same vocabulary, different input formatting. The effect is consistent (all 6 show compression) and the direction is meaningful (instruction formatting reduces spectral diversity).

However:

1. **The explanation is speculative.** Attributing this to "human alignment compressing semiotic geometry" assumes a framework (semiotic geometry, 8e as natural state) that is itself unproven.

2. **The effect may be trivial.** Adding prefixes like "query: " to inputs creates more similar embeddings (all start with the same tokens), which naturally reduces effective dimensionality and steepens eigenvalue decay. This is an input-formatting artifact, not evidence about "alignment compressing semiotic geometry."

3. **N=6 is small.** Six comparisons with no statistical test (no p-value, no confidence interval) is suggestive but not conclusive.

4. **The "natural state" claim is unfounded.** Calling plain-input Df * alpha the "natural state" and instruction-input Df * alpha the "distorted state" presupposes that 8e is fundamental. If 8e is not fundamental, then neither state is "natural" -- both are just different spectral statistics for different input distributions.

---

## Section 8: Internal Contradictions

### 8.1 Question Document vs. Report

The question document (q50_completing_8e.md) and the report (Q50_COMPLETING_8E.md) contradict each other on the Riemann connection:
- Question document (line 104): "The connection is analogous, not identical"
- Report (line 129): "This is not analogy. This is identity"

### 8.2 GLOSSARY vs. Report on alpha

The GLOSSARY (line 99) correctly warns: "This alpha has NO relation to the fine structure constant (alpha ~ 1/137)." But the report still describes alpha = 1/2 as "the Riemann critical line," which -- while not the fine structure constant -- still implies a deep connection to number theory that is not established.

### 8.3 SPECIFICATION vs. Report on Conservation Law

The SPECIFICATION (line 79-81) correctly states: "EMPIRICAL OBSERVATION. The proposed identity C = 8e = 21.746 is a curve fit." The report calls it a "conservation law" and "topologically protected."

### 8.4 HONEST_FINAL_STATUS vs. Q50

The HONEST_FINAL_STATUS.md (from Q54 audit) explicitly labels 8e as "NUMEROLOGY" with 15% confidence. Q50 labels it RESOLVED with R=1920. These assessments are fundamentally incompatible.

---

## Section 9: Inherited Issues from Phases 1-3

| Phase | Issue | Impact on Q50 |
|-------|-------|---------------|
| P1 | 5+ incompatible E definitions | Q50 does not use E directly but inherits the framework's definitional chaos |
| P1 | All evidence synthetic | CRITICAL: All 24 models tested on curated word lists, not real-world corpora |
| P2 | Quantum interpretation falsified | Q50's QGT/Chern derivation depends on Q44 Born rule (synthetic only) |
| P3 | R numerically unstable | Q50's Df * alpha is more stable than R, but alpha fit is sensitive to the range of k used |
| P3 | Test fraud pattern | Q50's tests follow the same pattern: set loose thresholds, then report "PASS" |

---

## Section 10: What Q50 Gets Right

In fairness:

1. **The empirical observation is real.** Df * alpha clusters around 20-23 across multiple embedding models. This is a genuine observation about the spectral structure of transformer embeddings.

2. **The alignment distortion effect is interesting.** Input formatting changes the eigenvalue structure in a consistent direction. This could be useful for studying how different input distributions affect embedding geometry.

3. **The Riemann connection is honestly downgraded in the question document.** The question document correctly calls it "analogous, not identical." (The report overclaims, but the primary source is honest.)

4. **The HONEST_FINAL_STATUS.md exists.** The project has an internal skeptic that correctly identifies the core problems. This is commendable.

5. **The cross-model analysis is a reasonable approach.** Testing the same quantity across many models is a valid methodology, even if the interpretation as a "conservation law" is overclaimed.

---

## Final Assessment

Q50 presents a genuine empirical regularity (Df * alpha ~ 21-22 across transformer embedding models) but dramatically overclaims its significance through:

1. **Mislabeling:** Calling a 7% CV statistical regularity a "conservation law"
2. **Numerology:** Fitting the value 21.75 to 8e and attaching post-hoc significance to both 8 and e
3. **Circular derivation:** The Chern number derivation assumes the conclusion (CP^n manifold) to derive it (alpha = 1/2)
4. **Non-independent samples:** 24 models that share architecture, training data, and preprocessing are not 24 independent tests
5. **Inadequate null hypothesis:** The random matrix baseline uses i.i.d. Gaussian noise rather than structure-preserving randomization
6. **Unfalsifiable framing:** Models matching 8e confirm the law; models deviating confirm alignment distortion
7. **Internal contradictions:** The question document and report disagree on the Riemann connection; HONEST_FINAL_STATUS.md labels 8e as "numerology" while Q50 labels it "RESOLVED"

The most credible finding is the alignment distortion effect, which could be interesting in its own right without the 8e framework. The least credible claims are the Riemann connection ("this is identity, not analogy") and the Chern number derivation.

**Recommended status: EXPLORATORY.** The empirical observation is real. The theoretical framework (8e, Peirce, Riemann, Chern) is speculative narrative, not established science. R should be reduced from 1920 to 600-800.

---

## Appendix: Issue Tracker Additions

| ID | Issue | Severity | Source | Affects |
|----|-------|----------|--------|---------|
| P4-Q50-01 | "Conservation law" label for 7% CV regularity is a category error | CRITICAL | Q50 | Central claim |
| P4-Q50-02 | 24 models are not independent (shared architecture/data/normalization) | CRITICAL | Q50 Q3 | Universality claim |
| P4-Q50-03 | e-per-octant is tautological (acknowledged but not resolved) | HIGH | Q50 Q1 | Interpretation |
| P4-Q50-04 | Peirce mapping to PCs is inconsistent across models (Q9 admits this) | HIGH | Q50 Q5 | "Why 3?" claim |
| P4-Q50-05 | Chern number derivation assumes CP^n without justification | CRITICAL | Q50 Report | alpha = 1/2 derivation |
| P4-Q50-06 | Report contradicts question document on Riemann ("identity" vs "analogy") | HIGH | Q50 Report vs Q50 Question | Riemann claim |
| P4-Q50-07 | Report contradicts HONEST_FINAL_STATUS.md (RESOLVED vs NUMEROLOGY) | HIGH | Q50 vs HONEST_FINAL_STATUS | Framework credibility |
| P4-Q50-08 | Random matrix null baseline uses wrong distribution | HIGH | test_q50_training_dynamics.py | Training dynamics claim |
| P4-Q50-09 | "Code models" are actually text models run on code snippets | MEDIUM | test_q50_cross_modal.py | Cross-modal claim |
| P4-Q50-10 | "Vision models" tested on text descriptions, not images | MEDIUM | test_q50_cross_modal.py | Cross-modal claim |
| P4-Q50-11 | Unfalsifiable design: matches confirm law, deviations confirm alignment distortion | CRITICAL | Q50 framing | Scientific validity |
| P4-Q50-12 | No pre-registered predictions; all analysis is post-hoc | HIGH | Q50 methodology | All claims |
| P4-Q50-13 | 2*pi growth rate is ordinary exponential growth, not a Riemann connection | MEDIUM | Q50 Report | Riemann claim |
| P4-Q50-14 | Entropy per octant test (H/e = 0.70) FAILS but is not counted as falsification | MEDIUM | Q50 Q1 | e interpretation |
| P4-Q50-15 | Instruction-tuning effect may be trivial input-formatting artifact | MEDIUM | Q50 Q4 | Alignment claim |
