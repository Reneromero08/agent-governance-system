# Verdict: Q38 - Noether Conservation (R=1520)

## Summary Evaluation

```
Q38: Noether Conservation (R=1520)
- Claimed status: ANSWERED (Cross-Architecture Validated)
- Proof type: Numerical experiment (synthetic geodesics + SLERP on real embeddings)
- Logical soundness: INVALID
- Claims match evidence: OVERCLAIMED
- Dependencies satisfied: MISSING [Q32 meaning field dynamics unvalidated; Q43 QGT connection assumed not derived]
- Circular reasoning: DETECTED [see Issue 1 below]
- Post-hoc fitting: DETECTED [see Issue 2 below]
- Recommended status: EXPLORATORY (analogical framework, not physics)
- Confidence: HIGH
- Issues: Fundamental tautology in core claim; Noether's theorem applied metaphorically not rigorously; "cross-architecture validation" tests a mathematical identity of SLERP not a physical property of embeddings; no time evolution defined; nothing is actually "conserved" in the physical sense
```

---

## Detailed Analysis

### Issue 1 (CRITICAL): The Core Claim is a Tautology

The central claim is that "angular momentum |L| = |v| is conserved along geodesics on the unit sphere." This is trivially, analytically true for ANY Riemannian manifold with the stated Lagrangian L = (1/2)|v|^2. It is a theorem of differential geometry that geodesic motion conserves speed. This is not a discovery about embeddings, meaning, or "the semiosphere" -- it is a restatement of the definition of geodesic motion with affine parameterization.

The synthetic tests (Tests 1-6) verify that the `sphere_geodesic()` function correctly implements the analytic formula `x(t) = x0 cos(|v|t) + v_hat sin(|v|t)`. The CV = 10^-15 result is machine-precision confirmation that NumPy's trigonometric functions work. This tells us nothing about embeddings or meaning.

**Verdict on synthetic tests:** These tests validate the correctness of a numerical implementation of a known formula. They do not validate any empirical claim about language or meaning.

### Issue 2 (CRITICAL): Cross-Architecture "Validation" Tests SLERP, Not Embeddings

The "cross-architecture validation" does the following:
1. Takes two word embeddings from a model (e.g., GloVe "truth" and "beauty")
2. Normalizes them to the unit sphere
3. Constructs a SLERP interpolation between them (which is, by definition, the geodesic on the sphere)
4. Measures angular momentum conservation along this SLERP trajectory
5. Finds CV ~ 6e-7 (near machine precision) and declares "conservation validated"

**This is circular.** SLERP is defined as the geodesic on the unit sphere. Geodesics on the unit sphere conserve angular momentum by construction. The test is: "If I construct a geodesic, does it behave like a geodesic?" The answer is trivially yes, regardless of what embeddings are used as endpoints.

The fact that this works "across 5 architectures" is meaningless. It would work with any two unit vectors in any dimension. The embeddings play no role whatsoever in the result. You could replace GloVe vectors with random unit vectors and get the same CV ~ 6e-7.

The "69,000x separation" between geodesic and perturbed paths is equally uninformative: it measures the difference between a geodesic (which conserves |L| by definition) and a randomly perturbed non-geodesic (which does not). This ratio depends on the perturbation noise_scale parameter (0.1), not on any property of embeddings.

### Issue 3 (CRITICAL): No Time Evolution is Defined

Noether's theorem requires:
1. A physical system with defined dynamics (equations of motion)
2. A continuous symmetry of the action
3. A derived conserved current/charge

The framework here posits that "concepts follow geodesics on the embedding sphere." But:

- **There is no dynamics.** Embeddings are static vectors. There is no time evolution operator, no Hamiltonian, no equation of motion that governs how a concept moves from one point to another.
- **SLERP is not dynamics.** SLERP is a mathematical interpolation method chosen by the experimenters. It is not an observed trajectory. The test constructs geodesics by fiat and then checks if they are geodesics.
- **No empirical trajectory exists.** At no point does the analysis track how a concept actually changes over time (e.g., how word embeddings evolve during training, or how context vectors change during inference). The "trajectories" are synthetic interpolations, not observed data.

### Issue 4 (MAJOR): Noether's Theorem is Applied Analogically, Not Rigorously

For a rigorous application of Noether's theorem, one needs:

1. **A Lagrangian for the system.** The claimed Lagrangian L = (1/2)|v|^2 is asserted, not derived from any model of how meaning evolves. It is the simplest possible Lagrangian on a manifold. No justification is given for why meaning dynamics should follow this Lagrangian rather than any other.

2. **A continuous symmetry of the action.** The SO(d) symmetry of the sphere is a property of the sphere, not of the embedding space. Actual embedding spaces are NOT rotationally symmetric -- different directions carry different semantic information. The PCA eigenvalue spectrum (which other questions in this framework measure) explicitly shows that some directions are far more important than others, breaking SO(d) to at most a much smaller group.

3. **A variational principle.** There is no demonstration that actual semantic processes extremize the stated action. The only demonstration is that the analytically constructed geodesics extremize it -- which is tautological.

### Issue 5 (MAJOR): What is "Conserved" Has No Physical Meaning

In physics, "conservation" means a quantity that does not change as the system evolves in time. Here:

- There is no time evolution (see Issue 3).
- The "conserved quantity" (|L| = |v|) is the speed along a geodesic, which is constant by the definition of affine parameterization of geodesics.
- Nothing is being "conserved" in any non-trivial sense. The statement reduces to: "If you parameterize a great circle with constant speed, the speed is constant."

The claim that "speed is conserved along geodesics" is presented as an empirical finding. It is a theorem -- specifically, a trivial consequence of the geodesic equation in affine parameterization.

### Issue 6 (MODERATE): Overclaimed Interpretations

The report makes several interpretive leaps that are not supported by the evidence:

1. **"Truth flows freely; lies fight the geometry"** -- There is no evidence connecting geodesics to truth or non-geodesics to deception. This is a metaphor presented as a finding.

2. **"Lie detection via conservation violation"** -- The proposed `is_deceptive()` function tests whether a trajectory is geodesic. Since no actual trajectory of deceptive vs. truthful reasoning has been tested, this is speculative.

3. **"This is physics, not a model artifact"** -- The CV ~ 6e-7 result is a property of SLERP interpolation, which is a mathematical construction. It is neither physics nor a model artifact; it is a mathematical identity.

4. **"Meaning has inertia"** -- No measurement of inertia (resistance to change of motion) has been performed on any semantic system.

### Issue 7 (MODERATE): The Falsified Hypothesis is More Informative Than the Confirmed One

The report honestly documents that the initial hypothesis (scalar momentum conservation in a "flat" principal subspace) was falsified. The pivot to angular momentum conservation is presented as "finding the correct physics." However, this pivot is a retreat from a testable, non-trivial claim to a tautological one:

- Original claim: "Scalar momentum is conserved in the Df~22 principal subspace" -- This is testable and was falsified (CV = 0.83).
- Replacement claim: "Angular momentum is conserved along geodesics" -- This is tautologically true and unfalsifiable.

The falsification of the original claim is scientifically valuable. The replacement claim is not.

### Issue 8 (MINOR): Loop Test Results Contradict the Narrative

The cross-architecture receipt shows that the "analogy loop" test (stitching multiple SLERP segments together) FAILS conservation in all 5 architectures:

| Architecture | Loop CV | Conserved? |
|---|---|---|
| GloVe | 0.090 | No |
| Word2Vec | 0.076 | No |
| FastText | 0.080 | No |
| BERT | 0.285 | No |
| SentenceT | 0.114 | No |

This is expected: stitching geodesic segments does not produce a geodesic (the velocity is discontinuous at junctions). But it contradicts the interpretive claim that "meaning flows along geodesics" -- if concepts actually followed geodesics, a semantic path king->queen->woman->man should be smooth, not piecewise with kinks.

### Issue 9 (MINOR): Perturbed Trajectories Also "Pass" Conservation in Real Embedding Tests

In the cross-architecture receipt, the perturbed trajectories have CV values of ~0.03-0.05, which are below the stated threshold of 0.05. In the GloVe, Word2Vec, FastText, BERT, and SentenceTransformer perturbed tests, "pairs_conserved" is 2/2 in every case -- meaning the perturbed paths ALSO pass the conservation test. The "separation ratio" is computed from raw CV values, but by the stated criterion (CV < 0.05 = conserved), the negative control fails to discriminate. This undermines the entire signal-separation narrative.

---

## Dependencies Check

| Dependency | Status | Issue |
|---|---|---|
| Q32 (Meaning Field) | Assumed | M = log(R) is assumed to "live on the semiosphere" but this is a definition, not a validated physical law |
| Q43 (QGT) | Assumed | "Curved geometry proven" is cited but the curvature of the embedding space is irrelevant to the tautological geodesic result |
| Q3 (Scale invariance) | Assumed | Connection to rotational symmetry is asserted without derivation |

---

## What Would Be Needed for a Valid Claim

1. **Define actual dynamics.** Show that some semantic process (e.g., word embedding evolution during training, context vector evolution during inference, human concept drift over time) produces trajectories that are empirically close to geodesics.

2. **Test against non-geodesic alternatives.** Compare geodesic interpolation to observed trajectories and show that the observed paths are geodesic, not merely that constructed geodesics are geodesic.

3. **Derive the Lagrangian.** Show why L = (1/2)|v|^2 is the correct Lagrangian for semantic dynamics, rather than asserting it by analogy.

4. **Measure conservation on real trajectories.** Track actual temporal evolution of embeddings and measure whether |L| is conserved along those trajectories.

5. **Address SO(d) breaking.** The embedding space is NOT rotationally symmetric (eigenvalue spectra are highly non-uniform). The actual symmetry group is much smaller. Identify the true symmetry and derive the corresponding conserved quantity.

---

## Conclusion

Q38 presents a tautological result (geodesics conserve angular momentum by definition) as an empirical discovery about meaning and semantics. The "cross-architecture validation" tests the mathematical properties of SLERP interpolation, not any physical property of embeddings. Noether's theorem is invoked analogically, not rigorously: there is no dynamical system, no derived Lagrangian, no observed time evolution, and no empirically measured conservation. The interpretive claims (truth as geodesic, lie detection, semantic inertia) are metaphors unsupported by the evidence presented.

The honest portions of the work -- particularly the falsification of scalar momentum conservation and the documentation of what does NOT hold -- are scientifically valuable. But the headline claim ("The semiosphere obeys Noether conservation laws") is OVERCLAIMED to the point of being misleading.

**Recommended status: EXPLORATORY** -- An analogical framework has been proposed and the mathematical scaffolding is correct, but no empirical content about semantics has been established.
