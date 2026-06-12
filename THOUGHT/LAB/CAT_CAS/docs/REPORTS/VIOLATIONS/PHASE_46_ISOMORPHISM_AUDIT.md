# Phase 46 Isomorphism Audit — Session 3 (2026-06-02)

**Auditor**: New agent (replacing previous agent)
**Method**: Read each experiment, verify isomorphism structure, check sensor validity, confirm null models

---

## 46.1 — Protein Foldability = Winding Number: **VALID** ✅

**Canonical file**: `46_1_foldability_oracle.py` (replaces deprecated `46_1_protein_folding_oracle.py`)

**Isomorphism**: 1D chain point-gap winding number as thermodynamic frustration sensor
- W=0 → foldable (balanced hopping, low frustration)
- W≠0 → frustrated/misfolded (unbalanced hopping, steric clash)

**Sensor**: Cauchy Argument Principle on 1D chain Hamiltonian with frustration-weighted hopping

**Structural validity**: The isomorphism is structurally sound. Uniform sequences (Poly-A, Poly-R) have balanced forward/backward hopping rates (W=0), while frustrated sequences (GP-repeat, random) have unbalanced hopping due to large KD differences between adjacent residues (W≠0). The winding number measures the net spectral flow around the origin, which directly corresponds to thermodynamic frustration in the contact topology.

**Tape**: Genuine catalytic tape (record_operation + uncompute + verify with bytes_written check)

**Null model**: Poly-A (uniform, zero frustration) as trivial foldable baseline. GP-repeat and random sequences as frustrated null comparison.

**Gates**: 3 gates testing foldable vs frustrated discrimination (Poly-A W=0, GP-repeat W≠0, 10/10 random W≠0)

**Documentation issue**: The `MASTER_REPORT_PHASE_46.md` describes 46.1 as using "2D contact map IPR" but the canonical implementation uses "1D chain winding number". These are fundamentally different sensors. The master report needs updating to reflect the canonical implementation.

**Verdict**: Isomorphism holds. Winding number is a valid frustration sensor.

---

## 46.2 — Folding Pathway = Gamma Sweep: **WEAK ISOMORPHISM** ⚠️

**Isomorphism**: Folding pathway → gamma sweep showing IPR discrimination between foldable and misfolded sequences

**Sensor**: 2D contact map IPR at different dissipation strengths (gamma = 0.0, 0.2, 0.5, 1.0, 2.0)

**Structural validity**: The measurement is real — IPR does discriminate between structured (helix) and random contacts at certain gamma values. However, the isomorphism to "folding pathway" is weak. A gamma sweep is just a parameter sweep over dissipation strength, not a dynamical folding pathway. The experiment shows that at high gamma (strong dissipation), IPR discrimination is maximized, but this doesn't map to a temporal folding process.

**Tape**: **CEREMONIAL** — local CatalyticTape class, never XOR-modified, verify() always passes

**Null model**: gamma=0 as no-solvent baseline (weak null — gamma=0 is just one end of the sweep)

**Gates**: 2 gates testing baseline gap and IPR discrimination at gamma=2.0

**Verdict**: Valid measurement (IPR discriminates contact topology), but weak isomorphism to "folding pathway". The gamma sweep is a parameter study, not a dynamical process. **Ceremonial tape needs fixing.**

---

## 46.3 — Prion Contagion = Impurity Detection: **PARTIAL ISOMORPHISM** ⚠️

**Isomorphism**: Prion as topological impurity in protein lattice

**Sensor**: Lattice IPR of 20-protein chain with one prion seed at center

**Structural validity**: The experiment is HONEST about limitations. It builds a lattice of 20 proteins (each 10 residues), with the center protein being a GP-repeat prion and the rest being poly-A. It measures IPR of the full 200x200 lattice Hamiltonian at different inter-protein coupling strengths J.

**Key finding** (from experiment's own honest note): "The prion seed creates localized impurity states (IPR elevation at J=0). Inter-protein coupling spreads these states across the lattice (IPR drops). The prion is DETECTABLE via IPR but does not 'propagate' its winding number to neighbors in this model. Contagion requires dynamical coupling not captured here."

**Tape**: **CEREMONIAL** — local CatalyticTape class, never XOR-modified

**Null model**: Healthy poly-A lattice as trivial baseline

**Gates**: 2 gates testing impurity detection (IPR > 0.05 at J=0) and delocalization (IPR drops with coupling)

**Verdict**: PARTIAL isomorphism. The experiment detects the prion as a topological impurity (localized states → high IPR) but does NOT demonstrate actual contagion propagation. The title "Prion Contagion" overstates what the experiment shows. It should be "Prion Impurity Detection". **Ceremonial tape needs fixing.**

---

## 46.4 — Topological Genetic Code: **VALID STRUCTURE, WEAKENED CLAIM** ⚠️

**Isomorphism**: Genetic code = error-correcting code on 64D codon lattice with point mutation adjacency

**Sensor**: Spectral radius and winding number of 64x64 Hamiltonian with non-reciprocal hopping weighted by KD polarity differences

**Structural validity**: The isomorphism structure is VALID. The 64 codons form a genuine topological space where edges connect codons differing by one nucleotide (point mutation graph). The Hamiltonian encodes chemical gradient as non-reciprocal hopping (Hatano-Nelson pump), which is physically motivated. The spectral radius measures frustration in the code's error-correction topology.

**Claim**: "The SGC is the unique zero-energy topological ground state"

**Problem**: The `MASTER_REPORT_PHASE_46.md` states: "5/9 mitochondrial codes have LOWER spectral radius than SGC (13.83–14.08)". This means the SGC is NOT the unique optimum — mitochondrial codes are spectrally superior. The central claim is weakened.

**Tape**: **CEREMONIAL** — local CatalyticTape class, never XOR-modified

**Null model**: 10 random shuffled codes (good null model)

**Gates**: 3 gates testing SGC ground state, alien frustration, grid independence

**Verdict**: VALID isomorphism structure, but the central claim (SGC = unique optimum) is WEAKENED by mitochondrial counterexamples. The SGC is an outlier (spectral radius 14.63 vs random mean 192.85), but not the unique ground state. **Ceremonial tape needs fixing.**

---

## 46.5 — Neural Binding = Winding Number: **VALID** ✅ (VERIFIED Session 3)

**Isomorphism**: Consciousness = topological edge state in connectome

**Sensor**: Winding number and IPR of non-Hermitian Watts-Strogatz connectome under intact/lesioned/anesthetized conditions

**Structural validity**: Already verified in Session 3. The experiment uses proper non-reciprocal hopping (chiral pump), genuine lesioning (removing edges, not building different graph), and measures both W and IPR.

**Verification (Session 3)**: Intact W=-21, IPR=0.0386. Lesioned 20% W=-17 (topology survives). Anesthetized (scale=0.05) W=0, IPR=0.7443 (19.3x localization). All gates pass.

**Tape**: Fixed in Session 3 to use shared BennettHistoryTape with genuine XOR modification

**Null model**: Anesthetized connectome (scale=0.05) as decohered baseline

**Gates**: 3 gates testing intact non-trivial topology, lesion survival, anesthesia localization

**Verdict**: VALID isomorphism, VERIFIED. The IPR localization transition under anesthesia is a genuine topological phase change.

---

## 46.6 — Morphogenesis = Defect Annihilation: **VALID** ✅

**Isomorphism**: Organ fold = topological edge state from nematic defect annihilation

**Sensor**: 1D slice winding and IPR through defect cores in 2D epithelial lattice

**Structural validity**: The isomorphism is structurally sound. The experiment builds a 2D lattice with +1/2 and -1/2 nematic defects (topological charge ±1/2). It extracts a 1D slice through the defect cores and computes both winding number and IPR. The three states (flat, separated, annihilated) map to:
- Flat sheet: no defects, delocalized eigenstates (low IPR)
- Separated defects: two ±1/2 defects, 0D point-localized states at EP cores (high IPR)
- Annihilated scar: defects merged, 1D extended edge mode along the scar (intermediate IPR)

The IPR progression (flat < annihilated < separated) is physically meaningful and maps to the topological classification of defect states.

**Tape**: **CEREMONIAL** — local CatalyticTape class, never XOR-modified

**Null model**: Flat sheet (no defects) as trivial baseline

**Gates**: 4 gates testing IPR discrimination (flat < 0.15, separated > 0.5, annihilated in (0.15, 0.5), sep/flat > 3x)

**Verdict**: VALID isomorphism. The defect annihilation → edge mode is genuine nematic defect physics. **Ceremonial tape needs fixing.**

---

## Phase 46 Summary

| Exp | Claim | Isomorphism Quality | Verdict |
|-----|-------|---------------------|---------|
| 46.1 | Foldability = winding | **VALID** — W measures thermodynamic frustration | ✅ VERIFIED (structure) |
| 46.2 | Pathway = gamma sweep | **WEAK** — parameter sweep, not dynamical pathway | ⚠️ LOOSE |
| 46.3 | Prion = contagion | **PARTIAL** — detects impurity, no propagation | ⚠️ OVERSTATED |
| 46.4 | Genetic code = ground state | **VALID structure, WEAKENED claim** — mitochondrial codes are superior | ⚠️ CLAIM WEAKENED |
| 46.5 | Consciousness = edge state | **VALID** — IPR localization transition under anesthesia | ✅ VERIFIED (Session 3) |
| 46.6 | Morphogenesis = defect annihilation | **VALID** — genuine nematic defect physics | ✅ VERIFIED (structure) |

**Score**: 3/6 valid isomorphisms, 1 weak, 1 partial/overstated, 1 valid structure but weakened claim.

**Critical issues**:
1. **5/6 experiments have ceremonial tapes** (46.2, 46.3, 46.4, 46.6, and the deprecated 46.1 old version). Only 46.5 was fixed in Session 3.
2. **46.3 title overstates**: "Prion Contagion" should be "Prion Impurity Detection"
3. **46.4 claim weakened**: SGC is not the unique optimum per mitochondrial codes
4. **46.1 documentation mismatch**: Master report describes 2D contact map IPR but canonical implementation uses 1D chain winding

**No experiments are null results.** All measure real effects. The isomorphisms range from structurally sound (46.1, 46.5, 46.6) to weak/overstated (46.2, 46.3, 46.4).
