# Phase 5.6: Polytope / Positive-Geometry Hypothesis Roadmap

**Date:** 2026-06-08
**Status:** `PHASE5_6_POLYTOPE_HYPOTHESIS_ROADMAP_ADDED`
**Author:** DeepSeek Agent, per user directive

---

## 1. Why Phase 5.6 Exists

CAT_CAS Track A has now produced a confirmed chain of catalytic-tape evidence:

1. **Phase 3B** — `RELATIONAL_INVARIANT_CONFIRMED`: Four-snapshot probe shows restored tape alone is insufficient; answer-predictive relational carrier survives T1/T2 and restoration; destructive, random, shuffled, same-final-hash wrong-answer controls are rejected.

2. **Phase 4.3** — `PHASE4_3_RESIDUAL_CHANNEL_PASS`: 3B carrier compressed into .holo-style 2-bit residual tags; wrong/random/destructive residual controls rejected; tape restores.

3. **Phase 4.4A** — `PHASE4_4A_OPERATOR_GOE_PASS`: Catalytic residual/operator matrices produce GOE-like NN spacing (r ~ 0.5482) vs Poisson diagonal null (~ 0.3775) and shuffled operator null (~ 0.3916). Software/catalytic operator-matrix evidence only.

4. **Phase 4.5** — `PHASE4_5_HOLO_MINI_MODEL_PASS`: Tiny .holo mini-model with shared basis slots, rotation slots, residual tags, operator-statistic support, readable class decode. Wrong/random residual rejection. Tape restore 24/24.

5. **Phase 4.6** — `PHASE4_6_PUBLIC_HOLO_HARNESS_PASS`: Reusable CLI harness packaging all Track A modes.

The cumulative evidence suggests something more than row-by-row algorithmic accident: a **structured relational carrier** spanning invariants, residuals, operator statistics, and decodable .holo basis structure. Phase 5.6 asks whether this structure is better described as:

- **Hypothesis A (algorithmic):** Step-by-step reversible tape operations, each independently successful
- **Hypothesis B (geometric):** Navigation/projection of a higher-dimensional relational geometry whose boundary constraints determine answer-carrying invariants

Phase 5.6 is a test of Hypothesis B. It does NOT claim Hypothesis B is true. It roadmaps the experiment to find out.

---

## 2. Existing Evidence for the Hypothesis

| Phase | Result | Geometric Interpretation |
|-------|--------|-------------------------|
| 3B | Four-snapshot invariant separates catalytic from all nulls | Catalytic points occupy a distinct region in invariant space |
| 4.3 | Residual tags compress carrier without losing answer correlation | Lower-dimensional encoding preserves relational structure |
| 4.4A | Operator matrices show GOE-like spacing against nulls | Relational carrier produces structured eigenvalue statistics |
| 4.5 | Mini-model decodes from basis + residual + rotation slots | Geometry supports readable multi-layer architecture |
| 4.6 | Harness works across modes | Behavior is reproducible and systematic |

These are individually credible but not yet tested as a unified geometric structure. Phase 5.6 tests the unified hypothesis.

---

## 3. Critical Correction: Why Scalar-Strength-Only Geometry Is Invalid

In Phase 3B, catalytic rows have:

```
strength_t0 = 1.000
strength_t1 = 1.000
strength_t2 = 1.000
strength_t3 = 1.000
```

across ALL accepted catalytic cases. If the polytope were built from these four scalars alone, the catalytic hull would collapse into a degenerate point at (1, 1, 1, 1). That would be structurally uninformative — a tautological "hull" that contains itself and excludes nothing meaningful.

**The roadmap must explicitly reject this degenerate construction.** The correct feature space uses the full relational carrier coordinate vector defined in Section 4.

---

## 4. Correct Feature Space Definition

Each run (P3B row, P4 experiment) maps to a point vector:

```
point_i = [
    snapshot strengths (4),
    answer/restoration fields (5),
    residual coordinates (10),
    XOR/parity coordinates (k),
    Walsh-Hadamard coordinates (k+2),
    graph spectral coordinates (k+2),
    .holo basis coordinates (2k+3),
    operator statistics (6),
    correlation / MI features (4)
]
```

### Feature Families

**A. Snapshot Strengths (4)**
- `strength_t0`, `strength_t1`, `strength_t2`, `strength_t3`
- Source: Phase 3B invariant probe
- WARNING: Will be identical across catalytic rows. Included for completeness but must not dominate the geometry.

**B. Answer/Restoration Fields (5)**
- `answer_correct`, `answer_corr`
- `restored`, `restore_distance`
- `wrong_answer_distance`, `null_distance`
- Source: Phase 3B output

**C. Residual Channel Coordinates (10)**
- `residual_tag_0`, `residual_tag_1`, `residual_tag_2`, `residual_tag_3`
- `residual_magnitude`
- `residual_hamming_distance`
- `wrong_residual_distance`, `random_residual_distance`
- `residual_decode_correct`
- `residual_decode_distance`
- Source: Phase 4.3 output

**D. XOR/Parity Features (k)**
- `parity_component_0` ... `parity_component_k-1`
- `pairwise_xor_relation_0` ... `pairwise_xor_relation_k-1`
- `carrier_parity_signature`
- Source: computed from Phase 3B row data

**E. Walsh-Hadamard Features (k+2)**
- `walsh_component_0` ... `walsh_component_k-1`
- `walsh_energy`, `dominant_walsh_index`, `walsh_sparsity`
- Source: Hadamard transform of Phase 3B row data

**F. Graph Spectral Features (k+2)**
- `graph_eigenvalue_0` ... `graph_eigenvalue_k-1`
- `graph_spectral_gap`
- `edge_relation_energy`
- Source: spectral decomposition of problem/operator graph

**G. Holo Basis Coordinates (2k+3)**
- `holo_basis_slot_0` ... `holo_basis_slot_k-1`
- `holo_residual_slot_0` ... `holo_residual_slot_k-1`
- `layer_id`, `shared_basis_id`, `layer_individuality_score`
- Source: Phase 4.5 mini-model output

**H. Operator Statistics (6)**
- `operator_mean_r`, `operator_std_r`
- `delta_to_GOE`, `delta_to_Poisson`, `delta_to_shuffled`
- `local_spacing_features`
- Source: Phase 4.4A output

**I. Correlation / MI Features (4)**
- `carrier_answer_mi`
- `slot_slot_correlation`
- `residual_answer_correlation`
- `basis_residual_correlation`
- Source: computed across Phase 3B/4 data

### Missing Feature Policy
- Missing features (e.g., Walsh components not yet computed) marked as NaN
- All NaN rows excluded from hull construction
- Feature count per row reported in dataset schema
- Minimum viable feature space: A + B + C + H (25 features) for initial geometry construction

---

## 5. Subphases

### 5.6.0 Feature-Space Definition
**Goal:** Define schemas, column sources, and explicit dimensionality.
**Output:** `phase5_6/FEATURE_SPACE_SPEC.md`
**Labels:** `PHASE5_6_FEATURE_SPACE_DEFINED`, `PHASE5_6_SCALAR_STRENGTH_HULL_REJECTED`
**Gate:** Schema written, scalar-only hull explicitly flagged as degenerate.

### 5.6.1 Dataset Builder
**Goal:** Build unified CSV from Phase 3B + Phase 4 artifacts.
**Input:** P3B probe CSV, P4.3 residual results, P4.4A GOE results, P4.5 mini-model results, P4.6 harness output.
**Classes:** catalytic, destructive_write, random_reversible_write, random_answer, shuffled_schedule, same_final_hash_wrong_answer, wrong_residual, random_residual, destructive_residual, shuffled_operator_null, poisson_operator_null.
**Output:** `phase5_6/results/polytope_feature_dataset.csv`, `phase5_6/results/polytope_feature_schema.csv`
**Labels:** `PHASE5_6_DATASET_BUILT`
**Gate:** All rows have class/seeds/family ids, all features normalized or marked raw, missing values explicit.

### 5.6.2 Convex Hull / Geometry Builder
**Goal:** Build catalytic geometric bodies from full carrier-coordinate space.
**Implementation:** `session_scripts/phase5_6/polytope_hypothesis.c`
**Methods:**
1. Exact hull in selected 2D/3D/4D projections
2. PCA projection hulls
3. Random projection hulls
4. Minimum-volume enclosing simplex if full hull unstable
5. k-NN distance-to-catalytic-region as non-hull backup
**Output:** `phase5_6/results/polytope_hull_stats.csv`
**Labels:** `PHASE5_6_GEOMETRY_BUILDER_READY`
**Gate:** At least one geometry method produces a separable catalytic region.

### 5.6.3 Null Exclusion / Cluster Separation Test
**Goal:** Test whether catalytic points occupy a region distinct from all nulls.
**Tests:** catalytic vs each null class (9 pairs).
**Metrics:** separation margin, null inclusion count, FPR, FNR, centroid distance, NN class accuracy.
**Key control:** same-final-hash wrong-answer MUST remain outside catalytic region.
**Labels:** `PHASE5_6_NULL_EXCLUSION_{PASS|PARTIAL|FAILED}`
**Gate:** PASS requires FPR < 0.1 and all major nulls excluded.

### 5.6.4 Holographic Projection Test
**Goal:** Test whether lower-dimensional projections preserve or lose separation.
**Projections tested:**
- Full feature space
- PCA: 8D, 4D, 3D, 2D, 1D
- Random projection: 4D, 3D, 2D
- Selected physical: residual-only, Walsh-only, graph-spectral-only, .holo-only, operator-stat-only, snapshot-strength-only
**Expected:** snapshot-strength-only should be weak/degenerate; full space should separate best; projection loss should be measurable and coherent.
**Labels:** `PHASE5_6_PROJECTION_HIERARCHY_{PASS|PARTIAL|COLLAPSE}`
**Gate:** PASS requires coherent hierarchy where lower dimensionality loses information predictably.

### 5.6.5 Predictive Geometry Test
**Goal:** Predict unseen runs from geometry before reading classification.
**Training:** Existing 3B + P4 data.
**Test:** Held-out seeds, new problem families, new operator schedules, new residual perturbation levels.
**Rule:** Inside/near catalytic hull -> predict invariant passes; outside -> predict failure/null.
**Labels:** `PHASE5_6_PREDICTIVE_GEOMETRY_{PASS|PARTIAL|FAILED}`
**Gate:** PASS requires prediction accuracy above random/null baselines for held-out data.

### 5.6.6 Entropy / CPU-Load Geometry Expansion Test
**Goal:** Test whether CPU saturation expands/deforms the catalytic geometry.
**Conditions:** low, medium, high load on Phenom II cores.
**Hypothesis:** Higher load should expand or deform the catalytic geometry while preserving answer-predictive structure up to some threshold (not simply destroy the invariant).
**Metrics:** hull volume per load, catalytic compactness, null exclusion, invariant strength, operator GOE stats, residual deformation.
**Labels:** `PHASE5_6_ENTROPY_GEOMETRY_EXPANSION_{PASS|PARTIAL|NOISE_ONLY}` or `LOAD_DESTROYS_INVARIANT`
**Gate:** PASS requires larger/richer catalytic region under load while preserving null separation.

### 5.6.7 Residual Boundary Deformation Test
**Goal:** Connect .holo residual tags to local polytope boundary deformation.
**Input:** Phase 4.3 residual tags, controlled perturbations (small/medium/large), wrong/random/destructive residuals.
**Tests:** Perturb residual magnitude -> recompute point -> measure hull displacement, decode correctness, restoration.
**Hypothesis:** Residual perturbation magnitude should predict local geometric deformation and decode behavior.
**Labels:** `PHASE5_6_RESIDUAL_BOUNDARY_DEFORMATION_{PASS|PARTIAL|FAILED}`
**Gate:** PASS requires monotonic relationship between perturbation magnitude and geometric displacement.

### 5.6.8 Polytope Decision Gate
**Final labels:**
- `PHASE5_6_POLYTOPE_GEOMETRY_CONFIRMED` — catalytic region excludes major nulls, projection hierarchy coherent, unseen runs predicted above baseline, same-final-hash wrong-answer outside, residuals map to boundary
- `PHASE5_6_POLYTOPE_GEOMETRY_PARTIAL` — geometry separates nulls but prediction weak, or prediction works but hierarchy unclear
- `PHASE5_6_POLYTOPE_ANALOGY_WEAK` — only simple clustering, no prediction, no stable projection, no residual relationship
- `PHASE5_6_POLYTOPE_HYPOTHESIS_REJECTED` — catalytic and nulls overlap, prediction fails, same-final-hash wrong-answer accepted, residuals don't map

**Forbidden final labels:** AMPLITUHEDRON_PROVEN, COSMOLOGICAL_POLYTOPE_PROVEN, PHYSICAL_HOLOGRAPHY_PROVEN, QUANTUM_GEOMETRY_PROVEN.

---

## 6. Null Hierarchy

All null classes from Phase 3B + Phase 4 must be tested:

| Null | Type | Expected Geometric Behavior |
|------|------|---------------------------|
| destructive_write | P3B | Outside catalytic hull |
| random_reversible_write | P3B | Outside catalytic hull |
| random_answer | P3B | Outside catalytic hull |
| shuffled_schedule | P3B | Outside catalytic hull |
| same_final_hash_wrong_answer | P3B | **KEY CONTROL — must be outside** |
| wrong_residual | P4.3 | Outside catalytic hull |
| random_residual | P4.3 | Outside catalytic hull |
| destructive_residual | P4.3 | Outside catalytic hull |
| shuffled_operator_null | P4.4A | Outside catalytic hull |
| poisson_operator_null | P4.4A | Outside catalytic hull |

---

## 7. Projection Hierarchy

| Projection | Expected Behavior |
|------------|-------------------|
| Full feature space | Best separation — gold standard |
| PCA 8D | Good separation, slight degradation |
| PCA 4D | Moderate degradation |
| PCA 3D | Observable but weakened |
| PCA 2D | Marginal, separable clusters |
| PCA 1D | Weak or collapsed |
| Random 4D | Comparable to PCA 4D if structure is geometric |
| Random 3D | Observable |
| Random 2D | Marginal |
| Residual-only | Strong signal if residuals carry primary geometry |
| Walsh-only | Weak if Walsh is noise-dominant |
| Graph-spectral-only | Test — may carry structure |
| .holo-only | Test — basis structure may project well |
| Operator-stat-only | Moderate — carries statistical signal |
| Snapshot-strength-only | **Degenerate (expected FAIL)** |

---

## 8. Entropy/Load Geometry Test

| Load Level | Background Cores | Expected if Geometric |
|------------|-----------------|----------------------|
| Low | 0 background | Baseline hull, reference catalytic region |
| Medium | 1-2 background cores, moderate cache pressure | Slight hull expansion/deformation |
| High | 3-4 background cores, high cache contention | Observable hull expansion or deformation |
| Saturated | All non-isolated cores saturated | May expand or collapse — test boundary |

**Key claim boundary:** Do not claim thermodynamic entropy directly. Use operational labels: load state, contention level, timing jitter, cache pressure proxy.

---

## 9. Residual Boundary Deformation Test

| Perturbation Magnitude | Expected if Geometric |
|------------------------|----------------------|
| 0.0 (original) | Reference point, inside catalytic hull |
| Small (0.1-0.2) | Slight hull displacement, decode still correct |
| Medium (0.3-0.5) | Moderate displacement, decode may degrade |
| Large (0.6-1.0) | Large displacement, decode likely fails |
| Wrong residual | Outside hull, decode fails |
| Random residual | Outside hull, decode fails |
| Destructive | Outside hull, decode impossible |

---

## 10. Decision Gate

```
Build dataset (5.6.1)
    |
    v
Construct geometry (5.6.2)
    |
    v
Does catalytic region exclude nulls? (5.6.3)
    |
    +--- NO  ---> PHASE5_6_POLYTOPE_HYPOTHESIS_REJECTED
    |
    +--- PARTIALLY ---> continue to projections
    |
    +--- YES ---> continue to projections
    |
    v
Do projections lose info coherently? (5.6.4)
    |
    +--- NO  ---> PHASE5_6_POLYTOPE_ANALOGY_WEAK
    |
    +--- YES ---> continue to prediction
    |
    v
Does geometry predict unseen runs? (5.6.5)
    |
    +--- NO  ---> PHASE5_6_POLYTOPE_GEOMETRY_PARTIAL
    |
    +--- YES ---> continue to entropy/residual
    |
    v
Entropy expansion (5.6.6) + residual boundary (5.6.7)
    |
    v
PHASE5_6_POLYTOPE_GEOMETRY_CONFIRMED
```

---

## 11. Forbidden Claims (MANDATORY)

This phase does NOT claim:
- Actual amplituhedron
- Actual cosmological polytope
- Physical holography
- Physical Kuramoto synchronization
- Quantum coherence
- Physical GOE from silicon
- Landauer violation
- Zero heat
- Microscopic entropy reduction
- Physical limit violation of any kind

**This phase may claim only:** CAT_CAS Track A data is consistent with a polytope-like relational geometry — a computational analogue where the carrier-coordinate geometry excludes nulls and predicts unseen behavior. The strongest allowed statement is "data-supported polytope-like computational geometry," not "proven amplituhedron/physical holography."

---

## 12. Implementation Prompt Placeholder

```
PHASE 5.6 IMPLEMENTATION PROMPT PLACEHOLDER

Task: Implement Phase 5.6 subphases 5.6.0 through 5.6.8.

Do not implement until explicitly asked.
Do not run the harness until asked.
Do not close Phase 5.6 until the decision gate produces a final label.

When triggered, begin at 5.6.0 (Feature-Space Definition) and proceed sequentially.
Stop at the first FAILING gate and report.
```

---

## 13. Final Status

```
PHASE5_6_POLYTOPE_HYPOTHESIS_ROADMAP_ADDED
PHASE5_6_NOT_IMPLEMENTED
PHASE5_6_AWAITING_GO
```

Next: Await explicit implementation trigger. Do NOT start subphases automatically.
