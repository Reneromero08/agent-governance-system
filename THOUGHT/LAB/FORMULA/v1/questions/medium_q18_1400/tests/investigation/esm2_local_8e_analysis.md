# ESM-2 Local 8e Analysis: Does 8e Emerge in Functional Regions?

**Date:** 2026-01-25
**Status:** INVESTIGATION PROPOSAL
**Author:** Claude Opus 4.5
**Related:** protein_embeddings_8e_analysis.md, Q18_SYNTHESIS.md

---

## Executive Summary

Previous analysis showed that ESM-2 global embeddings exhibit Df x alpha = 45-52, significantly higher than the theoretical 8e = 21.75 (approximately 2x deviation). However, this was measured across all residues globally. This investigation proposes testing whether **8e might emerge in LOCAL regions** where functional meaning is most concentrated - specifically binding sites, active sites, conserved motifs, and functional domains.

**Core Hypothesis:** If 8e represents the "geometry of meaning," it should appear most strongly where proteins encode their most concentrated functional semantics.

---

## 1. Background and Motivation

### 1.1 Global ESM-2 Results (Previous Findings)

From `protein_embeddings_8e_results_v2.json`:

| Configuration | Df x alpha | Deviation from 8e |
|--------------|------------|-------------------|
| Per-residue (3000 samples) | 51.95 | 138.9% |
| Sliding window (50 res) | 15.91 | 26.8% |
| Random baseline | 88.09 | 305.1% |

**Key observations:**
- Global ESM-2 has Df approximately 40 with alpha approximately 1.3
- The product (approximately 52) is roughly 2.4x the 8e value
- Sliding window (local averaging) reduces to approximately 16, BELOW 8e
- Neither approaches 8e precisely

### 1.2 Why Local Regions Might Be Different

**Theoretical reasoning:**

1. **Semiotic Compression**: Functional sites encode maximal meaning in minimal sequence space. Evolution has compressed information at these sites.

2. **Conservation Pressure**: Active sites and binding interfaces are under strong selective pressure, potentially creating more structured semantic relationships.

3. **Context Specificity**: While global embeddings capture "average" sequence semantics, local functional regions capture domain-specific meaning that may better align with Peirce's semiotic categories.

4. **Signal vs Noise**: Non-functional regions (linkers, disordered regions) may dilute the 8e signal when computed globally.

### 1.3 The Sliding Window Clue

The sliding window result (Df x alpha = 15.91) is intriguing:
- It is LOWER than 8e, not higher
- It suggests local averaging reduces dimensionality
- But sliding windows average EVERYTHING, including non-functional regions

**What if we specifically target functional regions?**

---

## 2. Proteins and Their Functional Regions

### 2.1 TP53 (P04637) - 393 residues

The "guardian of the genome" - one of the best-characterized proteins.

| Region | Residues | Function | Expected 8e? |
|--------|----------|----------|--------------|
| Transactivation Domain 1 (TAD1) | 1-42 | Transcription activation | Medium |
| Transactivation Domain 2 (TAD2) | 43-62 | Transcription activation | Medium |
| Proline-Rich Region (PRR) | 64-92 | Regulatory/structural | Low |
| **DNA-Binding Domain (DBD)** | 102-292 | DNA recognition | **HIGH** |
| Loop L2 (DBD) | 164-194 | Direct DNA contact | **VERY HIGH** |
| Loop L3 (DBD) | 237-250 | Direct DNA contact | **VERY HIGH** |
| Tetramerization Domain | 325-356 | Oligomerization | Medium |
| Regulatory Domain (CTD) | 363-393 | Intrinsically disordered | **LOW (Control)** |

**Hot spot residues:** R175, G245, R248, R249, R273, R282 (most mutated in cancer)

Sources: [UniProt P04637](https://www.uniprot.org/uniprotkb/P04637/entry), [PubMed 8276238](https://pubmed.ncbi.nlm.nih.gov/8276238/)

### 2.2 BRCA1 (P38398) - 1863 residues

Tumor suppressor with well-defined functional domains.

| Region | Residues | Function | Expected 8e? |
|--------|----------|----------|--------------|
| **RING Domain** | 1-109 | E3 ubiquitin ligase | **HIGH** |
| RING Finger Motif | 24-65 | Zn2+ coordination | **VERY HIGH** |
| Nuclear Export Signal | 81-99 | Localization | Low |
| Central Region | 110-1279 | Less structured | **LOW (Control)** |
| Serine Cluster (SCD) | 1280-1524 | Phosphorylation sites | Medium |
| Coiled-Coil (CC) | 1364-1437 | Protein interaction | Medium |
| **BRCT Domain 1** | 1650-1736 | Phosphoprotein binding | **HIGH** |
| **BRCT Domain 2** | 1756-1855 | Phosphoprotein binding | **HIGH** |

**Key residues:** C61, C64 (RING), S1655, G1656, K1702 (BRCT phosphoserine recognition)

Sources: [PMC3380633](https://pmc.ncbi.nlm.nih.gov/articles/PMC3380633/), [PMC4550207](https://pmc.ncbi.nlm.nih.gov/articles/PMC4550207/)

### 2.3 EGFR (P00533) - 1210 residues

Receptor tyrosine kinase with therapeutic relevance.

| Region | Residues | Function | Expected 8e? |
|--------|----------|----------|--------------|
| Signal Peptide | 1-24 | Secretion | Low |
| Extracellular Domain | 25-645 | Ligand binding | Medium |
| Transmembrane | 646-668 | Membrane anchor | Low |
| Juxtamembrane | 669-712 | Regulatory | Medium |
| **Kinase Domain N-lobe** | 712-815 | ATP binding | **HIGH** |
| **ATP Binding Site** | 718-726 | Gly-rich loop | **VERY HIGH** |
| **alphaC Helix** | 753-767 | Catalytic regulation | **HIGH** |
| **Catalytic Loop** | 812-818 | Catalysis | **VERY HIGH** |
| **DFG Motif** | 831-833 | Activation state | **HIGH** |
| **Activation Loop** | 831-852 | Regulatory | **HIGH** |
| C-terminal Tail | 960-1210 | Autophosphorylation | Medium |

**Catalytic residues:** K745, E762 (catalytic pair), D812, D831, Cys797 (drug target)

Sources: [PMC8838133](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838133/), [PMC6441332](https://pmc.ncbi.nlm.nih.gov/articles/PMC6441332/)

### 2.4 Additional Proteins for Analysis

From our existing cache (`extended_plddt.json`):

| Protein | ID | Length | Key Functional Region |
|---------|-----|--------|----------------------|
| HRAS | P01112 | 189 | GTPase domain (entire protein) |
| SRC | P12931 | 536 | Kinase domain (270-523) |
| AKT1 | P31749 | 480 | Kinase domain (150-408) |
| CDK4 | P11802 | 303 | Kinase domain (3-295) |
| CASPASE-3 | P42574 | 277 | Active site (163-175) |
| MAPK1 (ERK2) | P27361 | 379 | Kinase domain (25-313) |

---

## 3. Methodology

### 3.1 Local Embedding Extraction

```
For each protein P with functional region [start, end]:
    1. Run full sequence through ESM-2
    2. Extract per-residue embeddings for the entire sequence
    3. Subset embeddings to ONLY residues in [start, end]
    4. These become samples for Df x alpha calculation
```

**Key difference from global analysis:** We only use residues from functional regions, not all residues.

### 3.2 Analysis Strategy

**Step 1: Per-Protein Local Analysis**
- Extract embeddings for each defined functional region
- Compute Df, alpha, and Df x alpha for each region
- Compare to global protein-level value

**Step 2: Pooled Functional Category Analysis**
- Pool embeddings across proteins by functional category:
  - "DNA/RNA binding sites" (TP53 DBD, etc.)
  - "Kinase active sites" (EGFR, SRC, AKT1, CDK4, etc.)
  - "Protein-protein interfaces" (BRCA1 RING, BRCT)
  - "Disordered regions" (TP53 CTD, BRCA1 central region)
- Compute Df x alpha for each category
- Test if functional categories approach 8e

**Step 3: Control Comparisons**
- Random residue subsets (same size as functional regions)
- Disordered regions (expected LOW structuring)
- Linker regions (expected RANDOM behavior)

### 3.3 Statistical Thresholds

Following Q18 conventions:
- **PASS:** Deviation from 8e < 15%
- **WEAK:** Deviation 15-30%
- **FAIL:** Deviation > 30%

### 3.4 Sample Size Considerations

From previous analysis, we know sample size affects Df x alpha:

| n_samples | Df x alpha (ESM-2) |
|-----------|-------------------|
| 100 | 111.89 |
| 300 | 85.19 |
| 500 | 46.45 |
| 1000 | 47.66 |
| 2000 | 47.81 |
| 3000 | 50.46 |
| 5000 | 45.29 |

**Implication:** Small functional regions (30-50 residues) may show high variance. We should:
1. Pool residues across multiple proteins with similar functions
2. Use bootstrap confidence intervals
3. Report effect of sample size explicitly

---

## 4. Theoretical Framework: Why 8e Might Emerge Locally

### 4.1 Semiotic Compression at Functional Sites

Peirce's categories (Firstness, Secondness, Thirdness) map to:
- **Firstness:** What the site IS (chemical properties)
- **Secondness:** How the site REACTS (with substrates, partners)
- **Thirdness:** What the site MEANS (biological function)

At functional sites, these three are maximally coupled. A mutation at an active site:
- Changes chemical properties (First)
- Alters reaction kinetics (Second)
- Changes biological meaning (Third)

This tight coupling creates semiotic structure that may exhibit 8e.

### 4.2 Information Density Hypothesis

| Region Type | Information Density | Expected Df x alpha |
|-------------|--------------------|--------------------|
| Active site | VERY HIGH | Near 8e? |
| Binding interface | HIGH | Near 8e? |
| Conserved motif | HIGH | Near 8e? |
| Linker/loop | LOW | Near random (14.5)? |
| Disordered | MINIMAL | Below 8e? |

**Prediction:** Df x alpha should correlate with functional importance.

### 4.3 Why Global Analysis Misses 8e

The global Df x alpha of approximately 52 may be explained by:
1. **Dilution:** Non-functional residues add noise
2. **Mixing:** Distinct functional categories blend
3. **Scale mismatch:** 8e may apply at the local functional unit, not the whole protein

Analogy: The speed of sound in a concert hall depends on the ROOM properties, not the average of all spaces in the building.

---

## 5. Expected Outcomes and Interpretation

### 5.1 Scenario A: 8e Emerges in Functional Regions

**If:** DNA-binding domains, kinase active sites show Df x alpha approximately 21.75 (< 15% deviation)
**Then:**
- 8e IS the geometry of functional meaning
- Global deviation is due to dilution with non-functional residues
- Validates the "semiotic compression" hypothesis

### 5.2 Scenario B: Functional Regions Show Different but Consistent Values

**If:** Different functional categories show consistent Df x alpha values different from 8e:
- Kinase domains: approximately 35
- DNA-binding: approximately 28
- etc.

**Then:**
- Domain-specific "semiotic constants" exist
- 8e may be specific to natural language, not protein language
- Opens question of what determines each domain's constant

### 5.3 Scenario C: No Pattern Emerges

**If:** Functional regions show same Df x alpha as global (approximately 52) or high variance

**Then:**
- ESM-2 embedding geometry is uniform across the protein
- 8e genuinely does not apply to protein semantics
- Supports the conclusion that 8e is language-specific

### 5.4 Scenario D: Disordered Regions Show 8e

**If:** Disordered regions (expected to be low-structure) show 8e

**Then:**
- Major re-evaluation needed
- Would suggest 8e is NOT about functional compression
- May indicate methodological artifact

---

## 6. Implementation Plan

### 6.1 Phase 1: Data Preparation

1. Load protein sequences from existing cache
2. Define functional region boundaries (from Section 2)
3. Create annotation file mapping UniProt ID -> functional regions

### 6.2 Phase 2: Embedding Extraction

1. Load ESM-2 model (facebook/esm2_t6_8M_UR50D)
2. For each protein:
   - Compute per-residue embeddings
   - Extract local embeddings for each functional region
   - Store with region annotations

### 6.3 Phase 3: Local Df x alpha Analysis

1. For each functional region:
   - Compute Df, alpha, and Df x alpha
   - Record sample size and region type
2. Pool by functional category
3. Compare to controls

### 6.4 Phase 4: Statistical Analysis

1. Bootstrap confidence intervals
2. Compare functional vs non-functional regions
3. Correlation analysis: Df x alpha vs conservation scores
4. Report all deviations from 8e

---

## 7. Technical Appendix: Functional Region Definitions

### 7.1 Region Types

| Type | Description | Expected Semiotic Structure |
|------|-------------|----------------------------|
| **DNA_BINDING** | Contacts nucleic acids | HIGH - direct functional readout |
| **ATP_BINDING** | Phosphate transfer chemistry | HIGH - catalytic meaning |
| **CATALYTIC** | Where chemistry happens | VERY HIGH - maximal information |
| **PROTEIN_INTERFACE** | Binds other proteins | HIGH - interaction semantics |
| **PHOSPHORYLATION_SITE** | Regulatory modification | MEDIUM - signal transduction |
| **METAL_BINDING** | Coordinates metal ions | MEDIUM - structural/catalytic |
| **DISORDERED** | No stable structure | LOW - minimal semantic content |
| **LINKER** | Connects domains | LOW - spacer function |

### 7.2 Detailed Region Coordinates

```json
{
  "P04637": {
    "name": "TP53",
    "length": 393,
    "regions": [
      {"name": "TAD1", "start": 1, "end": 42, "type": "TRANSACTIVATION"},
      {"name": "TAD2", "start": 43, "end": 62, "type": "TRANSACTIVATION"},
      {"name": "PRR", "start": 64, "end": 92, "type": "PROLINE_RICH"},
      {"name": "DBD", "start": 102, "end": 292, "type": "DNA_BINDING"},
      {"name": "L2", "start": 164, "end": 194, "type": "DNA_CONTACT"},
      {"name": "L3", "start": 237, "end": 250, "type": "DNA_CONTACT"},
      {"name": "TET", "start": 325, "end": 356, "type": "OLIGOMERIZATION"},
      {"name": "CTD", "start": 363, "end": 393, "type": "DISORDERED"}
    ]
  },
  "P38398": {
    "name": "BRCA1",
    "length": 1863,
    "regions": [
      {"name": "RING", "start": 1, "end": 109, "type": "PROTEIN_INTERFACE"},
      {"name": "RING_FINGER", "start": 24, "end": 65, "type": "METAL_BINDING"},
      {"name": "CENTRAL", "start": 200, "end": 1000, "type": "DISORDERED"},
      {"name": "SCD", "start": 1280, "end": 1524, "type": "PHOSPHORYLATION_SITE"},
      {"name": "CC", "start": 1364, "end": 1437, "type": "COILED_COIL"},
      {"name": "BRCT1", "start": 1650, "end": 1736, "type": "PROTEIN_INTERFACE"},
      {"name": "BRCT2", "start": 1756, "end": 1855, "type": "PROTEIN_INTERFACE"}
    ]
  },
  "P00533": {
    "name": "EGFR",
    "length": 1210,
    "regions": [
      {"name": "EXTRACELLULAR", "start": 25, "end": 645, "type": "LIGAND_BINDING"},
      {"name": "TM", "start": 646, "end": 668, "type": "TRANSMEMBRANE"},
      {"name": "KINASE_N", "start": 712, "end": 815, "type": "ATP_BINDING"},
      {"name": "GLY_LOOP", "start": 718, "end": 726, "type": "ATP_BINDING"},
      {"name": "ALPHAC", "start": 753, "end": 767, "type": "CATALYTIC"},
      {"name": "CATALYTIC", "start": 812, "end": 818, "type": "CATALYTIC"},
      {"name": "DFG", "start": 831, "end": 833, "type": "CATALYTIC"},
      {"name": "ACTIVATION_LOOP", "start": 831, "end": 852, "type": "CATALYTIC"},
      {"name": "C_TAIL", "start": 960, "end": 1210, "type": "PHOSPHORYLATION_SITE"}
    ]
  },
  "P01112": {
    "name": "HRAS",
    "length": 189,
    "regions": [
      {"name": "P_LOOP", "start": 10, "end": 17, "type": "ATP_BINDING"},
      {"name": "SWITCH_I", "start": 30, "end": 40, "type": "GTP_BINDING"},
      {"name": "SWITCH_II", "start": 60, "end": 76, "type": "GTP_BINDING"},
      {"name": "G_DOMAIN", "start": 1, "end": 166, "type": "GTPASE"}
    ]
  },
  "P12931": {
    "name": "SRC",
    "length": 536,
    "regions": [
      {"name": "SH3", "start": 88, "end": 143, "type": "PROTEIN_INTERFACE"},
      {"name": "SH2", "start": 151, "end": 248, "type": "PROTEIN_INTERFACE"},
      {"name": "KINASE", "start": 270, "end": 523, "type": "CATALYTIC"}
    ]
  },
  "P31749": {
    "name": "AKT1",
    "length": 480,
    "regions": [
      {"name": "PH", "start": 6, "end": 108, "type": "LIPID_BINDING"},
      {"name": "KINASE", "start": 150, "end": 408, "type": "CATALYTIC"}
    ]
  },
  "P11802": {
    "name": "CDK4",
    "length": 303,
    "regions": [
      {"name": "KINASE", "start": 3, "end": 295, "type": "CATALYTIC"}
    ]
  },
  "P42574": {
    "name": "CASPASE3",
    "length": 277,
    "regions": [
      {"name": "PRODOMAIN", "start": 1, "end": 28, "type": "REGULATORY"},
      {"name": "LARGE_SUBUNIT", "start": 29, "end": 175, "type": "CATALYTIC"},
      {"name": "ACTIVE_SITE", "start": 163, "end": 175, "type": "CATALYTIC"},
      {"name": "SMALL_SUBUNIT", "start": 176, "end": 277, "type": "CATALYTIC"}
    ]
  }
}
```

---

## 8. Questions This Investigation Addresses

### 8.1 Primary Questions

1. **Does 8e emerge in LOCAL patches of protein embeddings where functional meaning is concentrated?**
   - Test: Compare Df x alpha of functional regions to global value

2. **What distinguishes regions with different Df x alpha?**
   - Test: Correlate Df x alpha with region type (catalytic, binding, disordered)

3. **Is there a relationship between functional importance and Df x alpha?**
   - Test: Compare active sites vs linkers vs disordered regions

### 8.2 Secondary Questions

4. **Is the global Df x alpha (approximately 52) just a mixture effect?**
   - Test: Can we reconstruct global from weighted sum of local values?

5. **Do proteins with more defined function (enzymes) show different patterns than scaffolds?**
   - Test: Compare kinases (well-defined function) to adaptor proteins

6. **Does conservation score correlate with local Df x alpha?**
   - Test: Use residue-level conservation scores from multiple sequence alignments

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| Small sample sizes per region | Pool across proteins, use bootstrap |
| ESM-2 model limitations | Consider testing ESM-2-650M or ESMFold |
| Region boundary uncertainty | Test sensitivity to boundary shifts (+/- 5 residues) |

### 9.2 Interpretation Risks

| Risk | Mitigation |
|------|------------|
| Multiple comparisons | Apply Bonferroni correction |
| Confirmation bias | Pre-register expected outcomes |
| Overfitting | Reserve some proteins for validation |

---

## 10. Connection to Q18 Synthesis

This investigation directly addresses the open prediction from Q18_SYNTHESIS.md:

> "**ESM-2 protein embeddings show 8e** - Embed proteins, compute Df x alpha - Expected CV < 15% near 21.75"

Previous testing found global ESM-2 does NOT show 8e (138.9% deviation). This investigation asks whether the failure is due to:
1. Global averaging hiding local 8e
2. Non-functional regions diluting the signal
3. Genuine absence of 8e in protein semantics

**If local functional regions show 8e, the prediction is VALIDATED with refinement.**
**If they do not, the prediction is FALSIFIED for protein embeddings.**

---

## 11. Conclusion

This investigation proposes a targeted analysis of ESM-2 embeddings at functional regions rather than globally. The theoretical motivation is that semiotic compression should be maximal at sites where sequence encodes maximal functional meaning.

**Expected deliverables:**
1. Df x alpha values for 30+ functional regions across 8+ proteins
2. Comparison by functional category (catalytic, binding, disordered)
3. Statistical analysis of 8e proximity by region type
4. Definitive answer to whether 8e emerges locally in protein embeddings

**If 8e emerges locally but not globally, this validates both:**
- The 8e theory (it applies to concentrated semantic content)
- The protein folding investigation (ESM-2 does encode functional meaning)

---

## References

1. UniProt TP53 Entry: https://www.uniprot.org/uniprotkb/P04637/entry
2. p53 DNA-binding Domain: https://pubmed.ncbi.nlm.nih.gov/8276238/
3. BRCA1 Structure-Function: https://pmc.ncbi.nlm.nih.gov/articles/PMC3380633/
4. BRCA1 Domain Mutations: https://pmc.ncbi.nlm.nih.gov/articles/PMC4550207/
5. EGFR Kinase Inhibitors: https://pmc.ncbi.nlm.nih.gov/articles/PMC8838133/
6. EGFR Binding Site Characterization: https://pmc.ncbi.nlm.nih.gov/articles/PMC6441332/

---

*Investigation proposed: 2026-01-25*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
