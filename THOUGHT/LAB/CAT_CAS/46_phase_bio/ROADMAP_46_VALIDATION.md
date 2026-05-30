# PHASE 46 ROADMAP v2: THE TOPOLOGICAL PROTEOME (BIOLOGY)
## From Synthetic Proof-of-Concept to Biological Ground Truth

---

## 1. STATUS: WHAT WE HAVE

Phase 46 has established that topological invariants of non-Hermitian Hamiltonians
**discriminate biological states** in synthetic models. Six experiments, six
working sensors, zero hardcoded invariants.

| Experiment | Sensor | Discriminates | Model Scale |
|-----------|--------|---------------|-------------|
| 46.1 | 2D Contact Map IPR | Folded vs misfolded | L=15-45, synthetic contacts |
| 46.2 | IPR vs Gamma | Foldable maintains 1/L IPR | 3 sequences × 2 contacts × 5 γ |
| 46.3 | Lattice IPR | Prion impurity detection | 20 proteins, J=0-1.0 |
| 46.4 | Spectral Radius + W | SGC vs random codes | 64 codons, 10 random trials |
| 46.5 | Winding + IPR | Intact vs anesthetized | L=150 Watts-Strogatz |
| 46.6 | 1D Slice IPR | Flat vs defect vs annihilated | 30×30 epithelium |

**All sensors are live. No hardcoded invariants. All gates pass.**

---

## 2. THE GAP: WHAT'S MISSING

Phase 46 proves the sensors **can** discriminate. It has not yet proven they
discriminate **biological ground truth** — real proteins, real connectomes,
real morphogenesis. The gap is in five dimensions:

### 2.1 Real Protein Contact Maps (Exps 46.1, 46.2)

**Current**: Synthetic alpha-helix contacts at (i, i+3) and (i, i+4). Random
contacts at 30% density.

**Missing**: Real contact maps from the Protein Data Bank (PDB). A lysozyme
(129 residues), ubiquitin (76 residues), or myoglobin (153 residues) with
its actual 3D structure-derived contacts. Compare the IPR of the native
contact map against the IPR of the sequence with shuffled contacts.

**The test**: Does the native contact map produce lower IPR (more extended
eigenstates) than shuffled contacts for the SAME sequence? If yes, the
topological sensor captures biological fold information. If no, the model
is missing essential physics.

### 2.2 Real Protein Sequences (Exps 46.1, 46.2)

**Current**: Synthetic sequences (poly-A, REWKYD, GP). Three archetypes.

**Missing**: Real protein sequences from UniProt. A set of known foldable
globular proteins vs known intrinsically disordered proteins (IDPs). Does
the IPR distinguish between sequences that fold in nature and sequences that
don't, when both are placed on the same contact map?

**The test**: Take 20 known globular proteins and 20 known IDPs. For each,
build the 2D contact map Hamiltonian using the real sequence and a generic
compact contact map. Does mean IPR cleanly separate the two classes? What's
the ROC curve?

### 2.3 Real Connectomes (Exp 46.5)

**Current**: Synthetic Watts-Strogatz graph, L=150, p_rewire=0.15.

**Missing**: The C. elegans connectome (302 neurons, ~5000 chemical synapses,
~2000 gap junctions). Directional edges, known synaptic weights from electron
microscopy reconstruction. Or the Drosophila hemibrain connectome. Real graph
structure, real edge weights, real directionality.

**The test**: Does the winding number of the real connectome change under
simulated lesioning that mirrors known neurological deficits? Does the IPR
response to anesthesia (synaptic scaling) match the state transitions
observed in EEG/fMRI?

### 2.4 Real Morphogenesis Geometry (Exp 46.6)

**Current**: 30×30 synthetic epithelium with analytically-placed ±1/2 defects.

**Missing**: Real epithelial cell positions and orientations from light-sheet
microscopy of gastrulating embryos (Drosophila, zebrafish). Defect positions,
cell polarity vectors, and active stress distributions extracted from
experimental data.

**The test**: Does the 1D slice IPR at real defect annihilation sites show
the predicted 1D extended mode? Does the location of the extended mode
correlate with the actual 3D fold that forms at that position in the embryo?

### 2.5 Validation Against Known Biology (All Experiments)

**Current**: Hardening gates test internal consistency — grid independence,
parameter sweeps, sensor live-ness. No comparison against external biological
truth.

**Missing**: For every experiment, a gold-standard dataset:
- **46.1/46.2**: Known folded vs known disordered proteins
- **46.3**: Known prion sequences (PrP, Aβ, α-synuclein) vs non-aggregating controls
- **46.4**: Conservation analysis — is the SGC's spectral minimality preserved across
  mitochondrial and alternative genetic codes?
- **46.5**: Connectomes from organisms with known conscious/unconscious states
- **46.6**: Embryos with known morphogenetic defects (mutants where gastrulation fails)

---

## 3. THE PATH FORWARD

### Mandate 1: Real Protein Validation (46.1-46.2 Upgrade) — COMPLETE
- [x] Fetch PDB structures for 10 globular proteins (ubiquitin, lysozyme, myoglobin, BPTI, crambin, RNase A, CI2, lambda repressor, SH3, tenascin)
- [x] Extract real contact maps (Cα distance < 8Å)
- [x] Build 2D contact map Hamiltonians with real sequences
- [x] Compute IPR for native contacts vs shuffled contacts
- [x] Compute IPR for 10 IDP sequences with random contacts
- [x] Cross-class analysis: Cohen's d = 0.37 (partial separation)
- **Result (20+20 proteins, hardened)**: Native vs shuffled: p=3e-7, t=7.6 ***. Glob vs IDP (raw IPR): p=0.01, d=0.64, CI[0.20,1.28] — VALIDATED. IPR*L normalized: p<0.0001, d=2.03 — VALIDATED (large effect). Size-matched: p<0.0001, d=1.78 — VALIDATED. Native contacts are longer-range (29.8 vs L/3). Mandate 1 fully validated.
- **Script**: `46_1_protein_folding/validation_real_pdb.py`

### Mandate 2: Real Connectome (46.5 Upgrade)
- [ ] Load C. elegans connectome (public dataset, 302 nodes, directed weighted edges)
- [ ] Build non-Hermitian Hamiltonian with real synaptic weights and phases
- [ ] Compute W and IPR for intact connectome
- [ ] Lesion known interneuron classes (AIY, AIZ, RIA) and track W/IPR changes
- [ ] Simulate anesthetic states by scaling inhibitory/excitatory weights

### Mandate 3: Real Morphogenesis (46.6 Upgrade)
- [ ] Extract epithelial cell positions and polarity from gastrulation microscopy
- [ ] Identify ±1/2 defect positions from the nematic director field
- [ ] Build non-Hermitian Hamiltonian with real cell positions and active stress
- [ ] Compute 1D slice IPR at defect annihilation sites
- [ ] Correlate IPR-detected 1D extended modes with actual 3D tissue folds

### Mandate 4: Cross-Validation Baselines
- [ ] For each experiment, define a NULL MODEL (randomized but dimension-matched)
- [ ] Compute the topological invariant for the null model
- [ ] Report the signal-to-null ratio (how many standard deviations above null)
- [ ] Replace "PASS/FAIL" gate language with statistical effect sizes

### Mandate 5: Conservation Analysis (46.4 Extension)
- [ ] Test the SGC spectral minimality against:
  - Mitochondrial genetic codes (vertebrate, invertebrate, yeast)
  - Alternative nuclear codes (ciliate, echinoderm)
  - Random codes (expanded from 10 to 1000 permutations)
- [ ] If SGC is the global minimum, compute the p-value of its spectral radius
  against the random ensemble

---

## 4. THE EPISTEMOLOGICAL STANDARD

Phase 45 established that topological invariants ARE the proof for mathematical
problems — the winding number, Chern number, and Cauchy argument principle are
mathematically equivalent to the truth value of the conjecture.

Phase 46 must meet a different standard. Biology is not a theorem to be proven;
it is a physical system to be **modeled and validated**. The topological
invariants are not the truth — they are the **sensor**. The question is whether
the sensor readings correlate with known biological states.

The roadmap from here is: replace synthetic models with real data, compute
the invariants, measure the correlation with biological ground truth, and
report effect sizes. The sensor works in principle. Now prove it works in
practice.
