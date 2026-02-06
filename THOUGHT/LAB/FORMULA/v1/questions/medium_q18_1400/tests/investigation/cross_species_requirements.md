# Cross-Species R Transfer Test: Analysis and Requirements

**Date:** 2026-01-25
**Status:** COMPLETED - Real test conducted, **NO SIGNIFICANT CORRELATION FOUND**

---

## Executive Summary

The original cross-species test (r=0.828) was **CIRCULAR** because mouse data was synthetically generated from human data with 72.5% conservation built in.

We have now conducted a **VALID** cross-species test using:
- **Human expression:** GEO microarray data (GSE13904, etc.) - 405 unique genes
- **Mouse expression:** GEO microarray data (GSE3431, GSE9954) - 21,722 unique genes
- **Orthologs:** Same-name matching (valid for 1:1 orthologs) + Ensembl mappings

**Result: r = 0.054, p = 0.378 (NOT SIGNIFICANT)**

---

## Data Independence Verification

| Data Source | Origin | Independence |
|-------------|--------|--------------|
| Human expression | NCBI GEO (GSE13904, GSE32474, etc.) | Independent |
| Mouse expression | NCBI GEO (GSE3431, GSE9954) | Independent |
| Ortholog mapping | Same-name genes (official nomenclature) | Independent |

**No data is derived from another.** This is a valid test.

---

## Test Results

### Correlation Statistics

| Metric | Value |
|--------|-------|
| Matched ortholog pairs | 271 |
| Pearson correlation | **0.054** |
| Spearman correlation | **-0.001** |
| p-value (permutation) | **0.378** |
| Z-score | 0.92 |

### R Value Statistics

| Species | Mean R | Std R | Min R | Max R |
|---------|--------|-------|-------|-------|
| Human | 6.48 | 2.44 | 1.41 | 14.48 |
| Mouse | 3.01 | 1.91 | 0.26 | 8.78 |

Note: Human R values are systematically higher than mouse R values (mean 6.48 vs 3.01).

### Sample Ortholog Pairs

| Human Gene | Human R | Mouse Gene | Mouse R |
|------------|---------|------------|---------|
| ZIC5 | 5.30 | ZIC5 | 1.60 |
| HS6ST3 | 8.21 | HS6ST3 | 6.43 |
| SERPINB11 | 8.02 | SERPINB11 | 0.49 |
| TXNDC2 | 8.79 | TXNDC2 | 0.45 |
| PANX2 | 14.48 | PANX2 | 3.29 |

---

## Interpretation

### Why the Original Test Was Circular

```python
# ORIGINAL (FAKE) - Mouse data derived from human data
mouse_expression[i] = (
    conservation * human_expr[i] +  # 72.5% DIRECT COPY
    noise
)
```

This guaranteed correlation because mouse R was mathematically derived from human R.

### Why This Test Is Valid

1. **Human expression**: Downloaded from GEO FTP servers (real microarray data)
2. **Mouse expression**: Downloaded from different GEO datasets (independent experiments)
3. **Ortholog matching**: Uses gene symbol matching (HUGO/MGI nomenclature)
4. **No synthetic data**: All R values computed from real experimental measurements

### What the Results Mean

**r = 0.054 (NOT SIGNIFICANT)** indicates:

1. **R does NOT transfer across species** in raw expression data
2. The original r=0.828 was entirely an artifact of circular data generation
3. Gene-level expression patterns (as measured by R) are species-specific

### Caveats

1. **Sample size**: Only 271 matched genes (limited by human expression dataset)
2. **Platform differences**: Human and mouse arrays may have systematic biases
3. **Tissue matching**: Datasets come from different tissue types
4. **R computation**: Both use R = mean/std, but different normalization methods

---

## What Additional Data Would Improve This Test

### Ideal Test Design

| Requirement | What We Need | Current Status |
|-------------|--------------|----------------|
| Matched tissues | Same tissue type in both species | NOT MET (different datasets) |
| Same platform type | RNA-seq for both | PARTIAL (both microarray, different platforms) |
| Larger sample | >1000 matched orthologs | NOT MET (271 genes) |
| Multiple datasets | 3+ datasets per species | PARTIAL (2 mouse, 5 human) |

### Recommended Data Sources

1. **GTEx + Mouse ENCODE**: Same tissues (brain, heart, liver, etc.)
2. **Large ortholog database**: BioMart Ensembl (16,000+ 1:1 orthologs)
3. **RNA-seq data**: More comparable across species than microarray

### Future Test Protocol

```
1. Download GTEx bulk RNA-seq (human) for 5+ tissues
2. Download ENCODE mouse RNA-seq for matching tissues
3. Download complete Ensembl 1:1 ortholog list (BioMart)
4. Compute R = mean/std for each gene in each species
5. Match by tissue type AND ortholog relationship
6. Compute correlation for tissue-matched pairs
```

---

## Conclusion

**The real cross-species test shows NO significant R transfer (r=0.054, p=0.38).**

This confirms that:
1. The original r=0.828 was completely circular and meaningless
2. R values computed from raw expression do NOT transfer across species
3. Cross-species validation of R requires either:
   - Same tissue types and experimental conditions
   - Or embedding-based R (where 8e might emerge)

---

## Files Generated

| File | Description |
|------|-------------|
| `test_cross_species_real.py` | The valid cross-species test implementation |
| `cross_species_real_results.json` | Full results with statistics |
| `cache/mouse_expression_real.json` | Real mouse GEO expression data |
| `cache/probe_to_gene_gpl1261_mouse.json` | Mouse probe annotation |

---

*This analysis confirms that Q18 cross-species claims cannot be validated with current data.*
*The original r=0.828 was a test artifact, not biological evidence.*

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
