# Verdict: Q51 Complex Plane & Phase Recovery (R=1940)

```
Q51: Complex Plane & Phase Recovery (R=1940)
- Claimed status: ANSWERED (3/4 tests CONFIRMED, 1/4 PARTIAL)
- Proof type: Empirical (PCA projection + winding number computation)
- Logical soundness: FATAL GAPS (complex structure imposed by projection, not discovered in data)
- Claims match evidence: SEVERELY OVERCLAIMED (projection artifacts relabeled as intrinsic complex structure)
- Dependencies satisfied: MISSING [complex-valued embeddings; intrinsic Berry curvature; Q43 Berry phase=0 contradicts Q51 Berry phase claims]
- Circular reasoning: DETECTED [project to 2D -> measure 2D winding -> claim "complex plane exists"]
- Post-hoc fitting: DETECTED [multiple tries, bug fixes that change results dramatically, flexible thresholds]
- Recommended status: REFUTED (Try3 confirms: experiments 1-2 fail for ALL 19 models)
- Confidence: HIGH (that the overclaim assessment is correct)
- Issues: See detailed analysis below
```

---

## 0. Document Consistency Problem

Three different documents make contradictory claims about Q51's status:

| Document | Status | Score |
|----------|--------|-------|
| q51_complex_plane.md (main) | ANSWERED / CONFIRMED | 3/4 CONFIRMED |
| Q51_COMPLEX_PLANE_REPORT.md (v6 report) | UNDER INVESTIGATION (MIXED) | 5 CONFIRMED, 1 PARTIAL, 1 WEAK, 2 INCONCLUSIVE, 1 FALSIFIED |
| Try2 global consistency report | N/A | Zero-signature NOT supported, roots-of-unity evidence = 0/19 |
| Try3 stress test | N/A | Exp1 = 0/19 pass, Exp2 = 0/19 pass |

The main Q51 file still claims "ANSWERED" and "CONFIRMED" despite the v6 report downgrading to "UNDER INVESTIGATION" with a falsified test, and despite Try3 showing two decisive failures. The status label was never corrected. This is the suppress-FALSIFIED pattern identified in prior phases.

---

## 1. Phase Arithmetic: Is Complex-Valued R Meaningful?

### The Fundamental Problem

Embedding vectors are real-valued elements of R^d (d=384 or 768). They contain no imaginary component. Complex structure is introduced entirely by the researcher:

1. Take real embeddings in R^d
2. Apply PCA to reduce to 2 dimensions
3. Declare PC1 = real axis, PC2 = imaginary axis
4. Compute z = PC1 + i*PC2
5. Measure phases, winding numbers, etc.

This is not "discovering" complex structure. It is **imposing** complex structure by arbitrary identification of two principal components with the real and imaginary parts of a complex number. You could do the same with ANY two-dimensional projection of ANY high-dimensional data and get "complex structure."

### Try2 Results (Favorable)

Try2 tested phase arithmetic on 19,544 Google analogies across 19 models. Pass rates ranged from 68.0% to 91.2% with a threshold of pi/4 (~45 degrees). All 19 models passed.

### Try3 Results (Devastating)

Try3 asked the critical question: **does phase arithmetic survive random 2D projections, rather than just the PCA projection?**

Results across ALL 19 models:
- **Exp1 (random bases):** 0/19 models pass. Phase arithmetic requires PCA specifically.
- **Exp2 (shared lambda):** 0/19 models pass. No consistent complex multiplier exists.

This is the key falsification: if embeddings had intrinsic complex structure, it would be visible from ANY 2D projection, not just the one that maximizes variance. The fact that PCA is required proves the "phase" is a PCA artifact, not an intrinsic property.

### The pi/4 Threshold is Extremely Generous

The pass condition is |phase_error| < pi/4 = 45 degrees. For a circle divided into sectors, 45 degrees is 1/8 of the full circle. A random guess would pass with probability 1/4 (since the error is |diff| which is at most pi, so values < pi/4 occur with probability ~25% for uniform random). The 80-90% Try2 pass rates are above chance, but the threshold is lenient enough that modest correlation suffices.

### Verdict on Phase Arithmetic

Phase arithmetic in the PCA plane is a real empirical observation. But it is an observation about PCA projections, not about "complex-valued semiotic space." PCA captures the dominant variance directions; word analogies (king-queen, man-woman) are well-known to create parallelogram structures in embedding space. When you project a parallelogram to 2D, the angle-difference consistency follows from the parallelogram rule, not from complex multiplication. Try3's decisive failure on random bases confirms this interpretation.

---

## 2. Berry Holonomy: Contradiction with Q43

### Q43 Proved Berry Phase = 0 for Real Vectors

The Q43 verdict (Phase 3) established:

> "For real vectors: <psi| d|psi> = (1/2) d(<psi|psi>) = (1/2) d(1) = 0. Therefore: Standard Berry phase = 0 for real vectors."

This is a mathematical theorem, not an empirical observation. Real vector bundles have trivially zero Berry curvature because there is no imaginary part to generate it. The Berry curvature is the antisymmetric (imaginary) part of the Quantum Geometric Tensor, which is identically zero for real vectors.

### Q51 Claims Nonzero Berry Phase

Q51 claims "Berry Holonomy CONFIRMED, Q-score = 1.0000 (perfect)." How?

The original Q51 method acknowledged the Berry phase = 0 problem and switched to a **winding number in a 2D projection**:

```python
def berry_phase_winding(path):
    centered = path - path.mean(axis=0)
    U, S, Vt = svd(centered)
    proj_2d = centered @ Vt[:2].T
    z = proj_2d[:, 0] + 1j * proj_2d[:, 1]
    phase_diffs = angle(z[1:] / z[:-1])
    return sum(phase_diffs)
```

This is NOT Berry phase. This is the winding number of a 2D projection of a curve in high-dimensional space. The code:
1. Takes a loop of 3-4 embedding vectors
2. Projects them to 2D via local SVD
3. Maps to the complex plane
4. Computes the winding number around the origin

### Why the "Q-score = 1.0000" is Trivially True

The Try2 berry phase report reveals the truth: **ALL models show |delta_gamma| = 0.0000 for both global and local projections, and quant_score = 1.0000 for ALL 19 models and ALL 42 loops.**

A quantization score of exactly 1.0000 for ALL loops and ALL models is not evidence of deep structure -- it is evidence of a degenerate metric. The winding number of any closed polygon with 3-4 vertices projected to 2D that encloses the origin will be exactly +/-1 (winding number = +/-2*pi), and any polygon not enclosing the origin will have winding number exactly 0. Integer winding numbers are a topological invariant of 2D curves -- they say nothing about the embedding space being "complex."

The quantization score measures `1 - |gamma/(2*pi) - round(gamma/(2*pi))|`. For a 3-point closed loop (triangle), the winding number in 2D is ALWAYS an integer or very close to one. The score will ALWAYS be ~1.0. This is a property of the COMPUTATION (2D winding numbers are integers by definition), not a property of the DATA.

### Try3's Exp3 (Winding Under Distortion)

Try3 Exp3 tested whether winding numbers survive small nonlinear distortions (sign, tanh). Result: 19/19 models pass. But this is trivially expected -- winding number is a topological invariant that survives any continuous deformation that does not move the curve across the origin. Small perturbations (epsilon=0.01) will not change winding numbers. This "test" tests continuity, not complex structure.

### Verdict on Berry Holonomy

The "Berry holonomy" test does not compute Berry phase (which is zero for real vectors, as Q43 proved). It computes 2D winding numbers, which are integers by topological necessity. The perfect quantization score is guaranteed by the mathematics of winding numbers, not discovered in the data. The Q51 document's "CRITICAL BUG FIX" that switched from the spherical excess formula (which gave non-quantized values) to the winding number formula (which gives automatically quantized values) is a textbook case of changing the metric until you get the answer you want.

---

## 3. Zero Signature: Genuine Discovery or Forced?

### The Claim

Octant phases (assigned as 8th roots of unity by construction) sum to approximately zero. Since the 8th roots of unity sum to zero, this supposedly confirms octants ARE roots of unity.

### The Circular Reasoning

The test assigns phase `theta_k = (k + 0.5) * pi/4` to octant k, then checks if `Sum e^(i*theta_k)` is near zero. But the phases are **assigned by the researcher**, not measured. The 8th roots of unity are **imposed**, not discovered. The test then checks that these imposed phases, weighted by octant populations, sum to approximately zero.

For the sum to be zero, all that's required is that octant populations are approximately equal. The v6 report acknowledges non-uniform octant populations (chi-squared p < 0.05 for all models), so the actual populations are NOT uniform.

### Try2's Devastating Replication

The Try2 zero-signature test with 19 models and 905 words found:
- Mean |S|/n = 0.0904 (original Q51 reported 0.0206)
- Null expectation E[|S|/n] = 0.0295
- **Roots-of-unity evidence: 0/19 models** (zero out of nineteen)
- All 19 models returned "False" for roots-of-unity evidence

The Try2 test added proper controls: checking DFT harmonics (m=2, m=3) and comparing to null expectations. Not a single model showed evidence of roots-of-unity structure. The "zero signature" is simply weak cancellation from roughly balanced octant populations, not roots of unity.

### Try2's Own Adversarial Interpretation

The Try2 report honestly states:

> "|S|/n near zero only constrains the first Fourier component of the octant distribution; it does NOT imply discrete roots-of-unity clustering. Small |S|/n is consistent with uniform random phase assignments and with simple opposite-pair cancellation."

### Verdict on Zero Signature

The zero signature is an artifact of approximately balanced (but not uniform) octant populations, combined with the researcher's choice to assign phases at exact 8th-root positions. The 8th roots of unity were not found in the data; they were assigned to the data by fiat. Try2's 19-model replication found zero evidence for actual roots-of-unity structure.

---

## 4. PCA Methodology: 8th Roots by Construction

### The Construction

The 8 "octants" are defined by the signs of PC1, PC2, PC3: (+,+,+), (+,+,-), etc. There are 2^3 = 8 sign patterns, hence 8 octants. This is a purely combinatorial fact about sign patterns in 3 dimensions. It has nothing to do with 8th roots of unity, complex planes, or phase structure.

The mapping from octants to phases (`theta_k = (k + 0.5) * pi/4`) is arbitrary. You could equally choose 7 PCs (128 "sectors") or 4 PCs (16 "sectors"). The number 8 is forced by choosing to use exactly 3 principal components. The Q51 narrative takes this accidental number, identifies it with the 8th roots of unity, and builds an elaborate complex-plane interpretation around it.

### The Pinwheel Test Failed

The one test that checked whether octants actually MAP to phase sectors (the pinwheel test) FAILED:
- Mean Cramer's V = 0.27 (threshold was > 0.5)
- Mean diagonal rate = 13.0% (threshold was > 50%)
- Random expectation for diagonal rate would be 12.5% (1/8)

A diagonal rate of 13.0% versus random expectation of 12.5% is essentially indistinguishable from chance. The pinwheel test conclusively falsifies the claim that octants correspond to phase sectors. Yet the main Q51 document still reports this as "PARTIAL" rather than "FALSIFIED."

---

## 5. Try2 vs Try3: The Pattern of Shifting Goalposts

### Timeline of Attempts

| Version | Result | Response |
|---------|--------|----------|
| Original Q51 | 3/4 CONFIRMED | Declared "ANSWERED" |
| V6 report | 5 CONFIRMED, 1 PARTIAL, 1 WEAK, 2 INCONCLUSIVE, 1 FALSIFIED | Downgraded to "UNDER INVESTIGATION" but main file not updated |
| Try2 phase arithmetic | 19/19 pass | Looks good |
| Try2 zero signature | 0/19 roots-of-unity evidence | Devastating failure |
| Try2 berry phase | Perfect scores (trivially) | Degenerate metric |
| Try2 global consistency | Zero-sig NOT supported, roots-of-unity NOT supported | Honest assessment: only "projection-level regularities" supported |
| Try3 stress test | Exp1: 0/19, Exp2: 0/19, Exp3: 19/19, Exp4: 12/19 | Conclusion: "harder to dismiss" |

### The Try3 Conclusion is Misleading

Try3's conclusion says: "Two or more experiments pass for at least one model; intrinsic complex structure becomes harder to dismiss."

But the two passing experiments (Exp3: winding survives distortion, Exp4: complex probe) are the WEAKEST tests of intrinsic complex structure:

- **Exp3** tests that winding numbers survive epsilon=0.01 perturbations. This is trivially expected from topological stability of winding numbers. It does not test for complex structure at all.
- **Exp4** tests that a ridge regression predicting complex lambda from embedding differences outperforms a phase-only probe. This shows that magnitude information is useful (predicting a 2D target is easier than predicting a 1D phase), not that complex structure is intrinsic.

The two STRONG tests (Exp1: phase survives random bases, Exp2: shared complex lambda exists) both return 0/19 across ALL models. These are the tests that would actually demonstrate intrinsic complex structure. Both fail completely.

The decision rule ("two or more pass") is a goalpost that was set to allow a favorable conclusion even when the most important experiments fail. The pre-registration (which to its credit was filed before results) predicted all four experiments would fail. Two failed (the important ones), two passed (the trivial ones). The conclusion should have been: "The null hypothesis is supported for the critical experiments."

---

## 6. R=1940 -- Is This Justified?

R=1940 is the second-highest score in the entire project. Let us evaluate what this score buys:

### What Was Actually Established

1. PCA projection of embedding analogies to 2D shows phase-difference consistency (~80-90% at pi/4 threshold). This is a projection of the well-known analogy parallelogram structure, not complex arithmetic.

2. Octant populations are roughly balanced (but not uniform), causing their assigned phases to approximately cancel. This is a property of PCA axes being centered, not roots of unity.

3. 2D winding numbers are integers. This is a theorem of topology, not a discovery about embeddings.

### What Was NOT Established

1. Embeddings are NOT shadows of complex-valued space. Try3 Exp1-2 definitively show the "complex" structure exists only in the PCA plane, not in random projections.

2. Berry phase is NOT nonzero. Q43 proved it is exactly zero for real vectors. Q51's "Berry phase" is actually a winding number, which is a different mathematical quantity.

3. The 8th roots of unity are NOT present in the data. Try2 found 0/19 models with roots-of-unity evidence.

4. Octants do NOT correspond to phase sectors. The pinwheel test failed with near-chance performance (13.0% vs 12.5% random).

### R=1940 is Grossly Inflated

The actual content of Q51 is: "PCA projections of word analogies to 2D show angular structure consistent with the parallelogram rule." This is a modest observation about a well-known property of word embeddings (Mikolov et al. 2013). It does not warrant R=1940, the second-highest score in the project. The inflation comes from wrapping this observation in language about "complex planes," "Berry holonomy," "phase sectors," and "roots of unity" -- none of which are supported by the evidence.

---

## 7. The Contextual Phase Selection "Breakthrough" (Q51.5)

Section Q51.5 claims that adding context to embedding prompts ("king, in terms of gender") dramatically reduces phase error from 161.9 to 21.3 degrees. This is presented as evidence that "the model already knows the phases."

### The Mundane Explanation

When you embed "king, in terms of gender" and "queen, in terms of gender," you are biasing the encoder toward the gender-relevant subspace. The resulting embeddings will be closer to each other along the gender axis and further apart along other axes. This is called **prompt engineering** and is a well-known technique in NLP. It does not demonstrate that the model "knows the phases" of a complex-valued space. It demonstrates that adding context constrains the embedding to a specific subspace, which is exactly what sentence transformers are designed to do.

### Phase Error Reduction as Trivial

If you tell the model "think about gender" for a gender analogy, of course the phase in PCA-2D will align with the gender direction. You have explicitly told the model what axis to use. The "87% reduction in phase error" is an observation about context-dependent embedding, not about complex structure.

---

## 8. Cross-Reference: The V6 Report's Own Admissions

The Q51_COMPLEX_PLANE_REPORT.md (v6) deserves credit for significant honesty. It:

1. Correctly identified that Test 6 (Method Consistency) is **FALSIFIED**: "Using truly independent PC pairs (PC12 vs PC34, no shared components): Real data correlation: 0.03-0.17. Random data correlation: 0.06."

2. Correctly identified that Test 5 (Phase Stability) is **INCONCLUSIVE**: "Random data shows same 58x error ratio as real data."

3. Correctly identified that Test 8 (Level Repulsion) had impossible values (beta > 1).

4. Correctly identified that Test 9 (Semantic Coherence) had a broken threshold (F > 2.0 when null gives F ~10.8).

These honest admissions ALREADY undermine the "ANSWERED" status of the main Q51 file. The main file was never updated to reflect these corrections.

Most critically, Test 6's falsification directly contradicts the core claim: if phases represent intrinsic structure, they must appear in PC3-4, not just PC1-2. They do not. The v6 report's own honest interpretation: "The 'phases' are real but confined to the PC1-2 plane (the semantic 'shadow')" concedes that the phenomenon is a property of the first two principal components, not of the embedding space itself.

---

## Summary Verdict

| Criterion | Score | Notes |
|-----------|-------|-------|
| Logical soundness | FAIL | Complex structure imposed by projection choice, not discovered |
| Claims match evidence | FAIL | "ANSWERED"/"CONFIRMED" status contradicts own v6 report and Try2-3 results |
| Berry phase claim | FAIL | Q43 proves Berry phase=0 for real vectors; Q51 computes winding numbers instead |
| Zero signature claim | FAIL | 0/19 models show roots-of-unity evidence in Try2 |
| Phase arithmetic claim | PARTIAL | Real effect in PCA plane, but fails on random bases (0/19 in Try3 Exp1) |
| Pinwheel (octant=phase) | FAIL | 13.0% diagonal rate vs 12.5% random; Cramer's V = 0.27 |
| Berry holonomy claim | FAIL | Perfect Q-score is guaranteed by definition of 2D winding number |
| Try3 stress test | FAIL | 2/4 experiments fail (the important ones); 2/4 pass (the trivial ones) |
| R=1940 justified | NO | Observation about PCA parallelogram structure, not complex geometry |
| Internal consistency | FAIL | Main file says ANSWERED; v6 report says UNDER INVESTIGATION with FALSIFIED test |
| Intellectual honesty | MIXED | V6 report is admirably honest; main file and Try3 conclusion are not updated accordingly |

### Recommended Status: REFUTED

The central hypothesis -- "Real embeddings are shadows of a fundamentally complex-valued semiotic space" -- is refuted by the project's own experiments:

1. **Try3 Exp1-2:** The two tests that would confirm intrinsic complex structure fail for ALL 19 models.
2. **V6 Test 6:** Phase structure does not extend beyond PC1-2 (FALSIFIED by the project's own assessment).
3. **Try2 Zero Signature:** 0/19 models show roots-of-unity evidence.
4. **Q43:** Berry phase is mathematically zero for real vectors.

What remains is: PCA projections of word analogies to 2D show angular regularity. This is the parallelogram rule for word analogies (Mikolov et al. 2013), observable in the first two principal components. It is not complex geometry, not Berry holonomy, and not 8th roots of unity.

### Recommended R Score: REDUCE to ~200

The genuine empirical content -- phase-difference consistency in PCA-2D for word analogies -- is a modest observation about a known property of word embeddings, not the second most important finding in the project.
