# Q18 Real Gene Expression Data Sources

## Summary

Successfully fetched **2,500 genes** from **5 GEO datasets** totaling **988 samples**.

**R = mean_expression / std_expression** computed for each gene.

## What Worked

### GEO Series Matrix Files (NCBI)

**Status: SUCCESS**

The GEO FTP server provides pre-processed expression matrices in "Series Matrix" format:
- URL pattern: `https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{GSE_ID}/matrix/{GSE_ID}_series_matrix.txt.gz`
- Format: Tab-separated, gzip compressed
- Data: Normalized expression values (log2 or similar)

**Datasets successfully fetched:**

| Dataset | Samples | Genes | Mean R | Description |
|---------|---------|-------|--------|-------------|
| GSE13904 | 227 | 500 | 5.31 | Pediatric sepsis blood |
| GSE32474 | 174 | 500 | 15.09 | Cancer expression |
| GSE14407 | 24 | 500 | 1.22 | Tissue comparison |
| GSE36376 | 433 | 500 | 32.42 | Large cohort |
| GSE26440 | 130 | 500 | 4.41 | Disease expression |

## What Did NOT Work

### ARCHS4 API
**Status: FAILED**
- The ARCHS4 API endpoint (https://maayanlab.cloud/archs4/data/api) returns 404
- ARCHS4py Python package requires local HDF5 files (10+ GB)
- Gene search API exists but does not return expression values

### GTEx Portal API
**Status: FAILED**
- API endpoint structure has changed
- Requires authentication or specific query parameters

### Expression Atlas (EBI)
**Status: FAILED**
- Download URLs for TSV files return 404
- API may require different endpoint format

### GEO DataSet SOFT Files
**Status: FAILED (URL format)
- GDS files exist but URL structure is different than expected
- Some datasets have moved or been archived

### Human Protein Atlas
**Status: PARTIAL**
- API returns JSON but expression data structure varies
- Not reliable for bulk expression extraction

## Data Quality Notes

1. **Platform**: Primarily Affymetrix Human Genome microarrays
2. **Normalization**: Pre-normalized by GEO submitters (RMA, MAS5, etc.)
3. **Identifiers**: Probe IDs (e.g., "1007_s_at") not gene symbols
4. **R Computation**: R = mean(expression) / std(expression) across samples
5. **Filtering**: Genes with < 10 samples or R outside (0.1, 1000) excluded

## R Statistics (Combined Dataset)

```
Mean R:   11.69
Median R: 5.75
Std R:    13.19
Min R:    0.33
Max R:    53.36
P25 R:    2.16
P75 R:    16.39
```

## R Distribution

| R Range | Count | Percentage |
|---------|-------|------------|
| 0-1     | 190   | 7.6%       |
| 1-2     | 402   | 16.1%      |
| 2-5     | 521   | 20.8%      |
| 5-10    | 557   | 22.3%      |
| 10-20   | 300   | 12.0%      |
| 20-50   | 524   | 21.0%      |
| 50+     | 6     | 0.2%       |

## Interpretation for Q18

- **R > mean(R)**: 760 genes (30.4%) - High consistency, likely housekeeping/essential genes
- **R < mean(R)**: 1,740 genes (69.6%) - Variable expression, tissue-specific or condition-responsive

The R threshold (mean R = 11.69) identifies genes with consistently high expression relative to their variance. These genes are candidates for essential cellular functions.

## Files Generated

- `cache/gene_expression_sample.json` - Full dataset with R values
- `cache/expression_summary.json` - Summary statistics
- `cache/geo_combined_expression.json` - Combined GEO data

## Future Improvements

1. Map probe IDs to gene symbols using platform annotation files
2. Download larger datasets when bandwidth permits
3. Try programmatic access to GTEx with proper API keys
4. Use ARCHS4py with downloaded HDF5 files for richer metadata
