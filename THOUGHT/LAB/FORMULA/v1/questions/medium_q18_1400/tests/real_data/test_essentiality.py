#!/usr/bin/env python3
"""
Test R vs Gene Essentiality using REAL DepMap data.

This test examines whether R (expression consistency = mean/std) predicts
gene essentiality as measured by CRISPR knockout experiments.

CRITICAL: This is NOT circular because:
- R is computed from expression patterns (consistency across samples/tissues)
- Essentiality is from CRISPR knockout experiments (cell viability)
- These are INDEPENDENT measurements

Data sources:
- DepMap: CRISPR gene effect scores (essentiality)
- HPA/GTEx: Gene expression across tissues (to compute R)
"""

import json
import math
import urllib.request
import ssl
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Disable SSL verification for some servers
ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'


def load_depmap_essentiality() -> Dict:
    """Load DepMap essentiality data."""
    with open(CACHE_DIR / 'depmap_essentiality.json', 'r') as f:
        data = json.load(f)
    return data['genes']


def fetch_hpa_expression(gene_symbol: str) -> Optional[Dict]:
    """
    Fetch gene expression from Human Protein Atlas.
    Returns R value and expression statistics.
    """
    # HPA uses ENSEMBL IDs, so we need to search by gene name
    url = f"https://www.proteinatlas.org/api/search_download?search={gene_symbol}&format=json&columns=g,eg"

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Q18 Research Bot)',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8')
            # Search results
            if not content.strip():
                return None
    except Exception as e:
        return None

    return None


def fetch_gtex_expression(gene_symbol: str) -> Optional[Dict]:
    """
    Fetch gene expression from GTEx Portal API.
    Returns R value (mean/std across tissues).
    """
    # GTEx API endpoint for median gene expression by tissue
    url = f"https://gtexportal.org/api/v2/expression/medianGeneExpression?geneSymbol={gene_symbol}&datasetId=gtex_v8"

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Q18 Research Bot)',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

            if 'data' not in data or not data['data']:
                return None

            # Extract TPM values across tissues
            values = [t.get('median', 0) for t in data['data'] if t.get('median', 0) > 0]

            if len(values) < 3:
                return None

            mean_expr = sum(values) / len(values)
            variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
            std_expr = math.sqrt(variance) if variance > 0 else 0.0001

            R = mean_expr / std_expr if std_expr > 0.0001 else 0

            return {
                'mean_expr': mean_expr,
                'std_expr': std_expr,
                'R': R,
                'n_tissues': len(values)
            }
    except Exception as e:
        return None


def compute_r_from_affymetrix_data() -> Dict[str, Dict]:
    """
    Use Affymetrix probe-to-gene mapping to get R values for genes.

    This is a common mapping for HG-U133 Plus 2.0 arrays (used in most GEO data).
    """
    # Standard Affymetrix probe -> gene mappings (curated subset)
    # These are well-known probe-gene mappings for HG-U133 Plus 2.0
    PROBE_TO_GENE = {
        # Housekeeping genes (expected high R)
        '1007_s_at': 'DDR1',
        '1053_at': 'RFC2',
        '117_at': 'HSPA6',
        '121_at': 'PAX8',
        '200000_s_at': 'PRPF18',
        '200001_at': 'CAPNS1',
        '200002_at': 'RPL35',
        '200003_s_at': 'RPL28',
        '200004_at': 'EIF4G2',
        '200005_at': 'EIF3D',
        '200006_at': 'PARK7',
        '200007_at': 'SRP14',
        '200008_s_at': 'GDI2',
        '200009_at': 'GDI2',
        '200010_at': 'RPL11',
        '200011_s_at': 'TTC3',
        '200012_x_at': 'RPL21',
        '200013_at': 'RPL24',
        '200014_s_at': 'HNRNPC',
        '200015_s_at': 'HNRNPC',
        '200016_x_at': 'RPS15A',
        '200017_at': 'RPL10A',
        '200018_at': 'RPS13',
        '200019_s_at': 'FAU',
        '200020_at': 'TARDBP',
        '200021_at': 'USP33',
        '200022_at': 'HDLBP',
        '200023_s_at': 'EIF3A',
        '200024_at': 'PUM1',
        '200025_s_at': 'RPL27A',
        '200026_at': 'RPL34',
        '200027_at': 'NACA',
        '200028_s_at': 'STARD7',
        '200029_at': 'RPL19',
        '200030_s_at': 'KIAA0100',
        '200031_s_at': 'SET',
        '200032_s_at': 'RPL9',
        '200033_at': 'DDX5',
        '200034_s_at': 'RPL6',
        '200035_at': 'NACA2',
        '200036_s_at': 'RPL10',
        '200037_s_at': 'ACTR2',
        '200038_s_at': 'RPL17',
        '200039_s_at': 'PSMB2',
        '200040_at': 'KHSRP',
        '200041_s_at': 'BAT1',
        '200042_at': 'C14orf2',
        '200043_at': 'ERH',
        '200044_at': 'SFRS9',
        '200045_at': 'ABCF1',
        '200046_at': 'DAD1',
        '200047_s_at': 'YY1',
        '200048_s_at': 'ILK',
        '200049_at': 'TMED2',
        '200050_at': 'ZDHHC5',
        # Cancer genes
        '201746_at': 'TP53',
        '211300_s_at': 'TP53',
        '201283_s_at': 'KRAS',
        '214702_at': 'EGFR',
        '201884_at': 'CEACAM5',
        '201667_at': 'GJA1',
        '200795_at': 'SPARCL1',
        '200853_at': 'RPS2',
        '200854_at': 'RPS11',
        '201790_s_at': 'DHCR7',
        '201389_at': 'ITGA5',
        '200018_at': 'RPS13',
        # Essential genes (from literature)
        '200685_at': 'ACTB',  # Beta-actin
        '200738_s_at': 'PGK1',  # Phosphoglycerate kinase
        '200041_s_at': 'BAT1',  # HLA-B associated transcript
        '201125_s_at': 'ANXA2',  # Annexin A2
        '201091_s_at': 'CBX3',  # Chromobox 3
    }

    # Load GEO expression data
    try:
        with open(CACHE_DIR / 'gene_expression_sample.json', 'r') as f:
            expr_data = json.load(f)
    except FileNotFoundError:
        return {}

    # Map probe R values to gene symbols
    gene_r_values = {}

    for probe_key, probe_data in expr_data.get('genes', {}).items():
        # Extract probe ID from key (format: GSE13904:1007_s_at)
        if ':' in probe_key:
            probe_id = probe_key.split(':')[1]
        else:
            probe_id = probe_key

        if probe_id in PROBE_TO_GENE:
            gene_symbol = PROBE_TO_GENE[probe_id]

            # Keep highest R if multiple probes map to same gene
            if gene_symbol not in gene_r_values or probe_data['R'] > gene_r_values[gene_symbol]['R']:
                gene_r_values[gene_symbol] = {
                    'R': probe_data['R'],
                    'mean_expr': probe_data['mean_expr'],
                    'std_expr': probe_data['std_expr'],
                    'probe_id': probe_id
                }

    return gene_r_values


def fetch_expression_batch(genes: List[str], max_genes: int = 200) -> Dict[str, Dict]:
    """
    Fetch expression data for a batch of genes from GTEx.
    Rate-limited to avoid API issues.
    """
    results = {}
    genes_to_fetch = genes[:max_genes]

    print(f"Fetching expression data for {len(genes_to_fetch)} genes from GTEx...")

    for i, gene in enumerate(genes_to_fetch):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(genes_to_fetch)}")

        expr = fetch_gtex_expression(gene)
        if expr:
            results[gene] = expr

        # Rate limiting
        time.sleep(0.1)

    print(f"  Got expression data for {len(results)} genes")
    return results


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 3:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Covariance
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

    # Standard deviations
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    return cov / (std_x * std_y)


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 3:
        return 0.0

    # Rank the values
    def rank(values):
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for rank_val, idx in enumerate(sorted_indices, 1):
            ranks[idx] = rank_val
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)

    return pearson_correlation(rank_x, rank_y)


def compute_auc(labels: List[bool], scores: List[float]) -> float:
    """
    Compute AUC for binary classification.
    Higher score should predict True label.
    """
    if not labels or len(labels) != len(scores):
        return 0.5

    # Sort by score descending
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: -x[0])

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count concordant pairs
    concordant = 0
    sum_ranks_positive = 0

    for rank, (score, label) in enumerate(pairs, 1):
        if label:
            sum_ranks_positive += rank

    # Mann-Whitney U statistic
    u = sum_ranks_positive - n_pos * (n_pos + 1) / 2
    auc = u / (n_pos * n_neg)

    return auc


def analyze_r_vs_essentiality():
    """
    Main analysis: Test if R predicts gene essentiality.

    Hypothesis: Genes with higher R (more consistent expression)
    should be more essential.
    """
    print("=" * 70)
    print("Q18: Testing R vs Gene Essentiality")
    print("=" * 70)

    # Load DepMap essentiality data
    print("\n1. Loading DepMap essentiality data...")
    depmap = load_depmap_essentiality()
    print(f"   Loaded {len(depmap)} genes with essentiality scores")

    # Get essential and non-essential counts
    n_essential = sum(1 for g in depmap.values() if g['essential'])
    n_nonessential = len(depmap) - n_essential
    print(f"   Essential (effect < -0.5): {n_essential}")
    print(f"   Non-essential: {n_nonessential}")

    # Get R values from existing expression data
    print("\n2. Getting R values from expression data...")
    gene_r = compute_r_from_affymetrix_data()
    print(f"   Got R values for {len(gene_r)} genes from GEO data")

    # If we have few genes from probe mapping, fetch from GTEx
    if len(gene_r) < 100:
        print("\n   Fetching additional expression data from GTEx API...")

        # Get essential genes to prioritize
        essential_genes = [g for g, d in depmap.items() if d['essential']][:100]
        nonessential_genes = [g for g, d in depmap.items() if not d['essential']][:100]

        genes_to_fetch = essential_genes + nonessential_genes

        gtex_data = fetch_expression_batch(genes_to_fetch, max_genes=200)

        for gene, expr in gtex_data.items():
            if gene not in gene_r:
                gene_r[gene] = expr

        print(f"   Total genes with R values: {len(gene_r)}")

    # Match genes between datasets
    print("\n3. Matching genes between datasets...")
    matched_genes = []

    for gene, r_data in gene_r.items():
        if gene in depmap:
            matched_genes.append({
                'gene': gene,
                'R': r_data['R'],
                'mean_effect': depmap[gene]['mean_effect'],
                'essential': depmap[gene]['essential']
            })

    print(f"   Matched {len(matched_genes)} genes")

    if len(matched_genes) < 10:
        print("\n   WARNING: Too few matched genes for reliable analysis")
        # Try with more relaxed matching or additional sources

    # Compute correlations
    print("\n4. Computing correlations...")

    r_values = [g['R'] for g in matched_genes]
    effects = [g['mean_effect'] for g in matched_genes]
    essentials = [g['essential'] for g in matched_genes]

    # Correlation between R and essentiality effect
    # More negative effect = more essential
    # Hypothesis: Higher R -> More negative effect (more essential)
    pearson_r = pearson_correlation(r_values, effects)
    spearman_r = spearman_correlation(r_values, effects)

    print(f"   Pearson correlation (R vs effect): {pearson_r:.4f}")
    print(f"   Spearman correlation (R vs effect): {spearman_r:.4f}")

    # Note: negative correlation means higher R -> more negative effect (more essential)
    # This would SUPPORT the hypothesis

    # Compute AUC for essential vs non-essential classification
    print("\n5. Computing AUC for essentiality classification...")

    # Higher R should predict essential (True)
    auc = compute_auc(essentials, r_values)
    print(f"   AUC (R predicts essential): {auc:.4f}")

    # Compare R distributions
    print("\n6. Comparing R distributions...")

    essential_r = [g['R'] for g in matched_genes if g['essential']]
    nonessential_r = [g['R'] for g in matched_genes if not g['essential']]

    if essential_r:
        mean_r_essential = sum(essential_r) / len(essential_r)
        print(f"   Mean R for essential genes: {mean_r_essential:.4f} (n={len(essential_r)})")
    else:
        mean_r_essential = 0
        print("   No essential genes in matched set")

    if nonessential_r:
        mean_r_nonessential = sum(nonessential_r) / len(nonessential_r)
        print(f"   Mean R for non-essential genes: {mean_r_nonessential:.4f} (n={len(nonessential_r)})")
    else:
        mean_r_nonessential = 0
        print("   No non-essential genes in matched set")

    # Effect size
    r_difference = mean_r_essential - mean_r_nonessential
    print(f"   Difference: {r_difference:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # For the correlation: negative means higher R -> more essential
    # (since essentiality effect is negative for essential genes)
    if pearson_r < -0.1:
        correlation_interpretation = "SUPPORTS hypothesis: Higher R correlates with essentiality"
    elif pearson_r > 0.1:
        correlation_interpretation = "CONTRADICTS hypothesis: Higher R correlates with non-essentiality"
    else:
        correlation_interpretation = "WEAK/NO relationship: R does not strongly predict essentiality"

    print(f"Correlation: {correlation_interpretation}")

    # For AUC: > 0.5 means higher R predicts essential
    if auc > 0.6:
        auc_interpretation = "SUPPORTS hypothesis: R has predictive power for essentiality"
    elif auc < 0.4:
        auc_interpretation = "CONTRADICTS hypothesis: Lower R predicts essentiality"
    else:
        auc_interpretation = "WEAK predictive power: R marginally predicts essentiality"

    print(f"AUC: {auc_interpretation}")

    # Overall conclusion
    if mean_r_essential > mean_r_nonessential and auc > 0.55:
        conclusion = "HYPOTHESIS SUPPORTED: Essential genes tend to have higher R"
    elif mean_r_essential < mean_r_nonessential and auc < 0.45:
        conclusion = "HYPOTHESIS REJECTED: Essential genes tend to have lower R"
    else:
        conclusion = "INCONCLUSIVE: No clear relationship between R and essentiality"

    print(f"\nCONCLUSION: {conclusion}")

    # Prepare results
    results = {
        'analysis': 'R vs Gene Essentiality',
        'data_sources': {
            'essentiality': 'DepMap CRISPR gene effect scores',
            'expression_r': 'GEO Series Matrix + GTEx API'
        },
        'sample_sizes': {
            'total_depmap_genes': len(depmap),
            'total_expression_genes': len(gene_r),
            'matched_genes': len(matched_genes),
            'essential_matched': len(essential_r),
            'nonessential_matched': len(nonessential_r)
        },
        'statistics': {
            'pearson_correlation': round(pearson_r, 4),
            'spearman_correlation': round(spearman_r, 4),
            'auc_essential_classification': round(auc, 4),
            'mean_r_essential': round(mean_r_essential, 4),
            'mean_r_nonessential': round(mean_r_nonessential, 4),
            'r_difference': round(r_difference, 4)
        },
        'interpretation': {
            'correlation': correlation_interpretation,
            'auc': auc_interpretation,
            'conclusion': conclusion
        },
        'validation': {
            'is_circular': False,
            'reason': 'R from expression patterns, essentiality from CRISPR knockouts - independent measurements'
        },
        'matched_genes_sample': matched_genes[:20]  # First 20 for inspection
    }

    # Save results
    output_file = Path(__file__).parent / 'essentiality_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    results = analyze_r_vs_essentiality()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Matched genes: {results['sample_sizes']['matched_genes']}")
    print(f"Pearson r: {results['statistics']['pearson_correlation']}")
    print(f"Spearman r: {results['statistics']['spearman_correlation']}")
    print(f"AUC: {results['statistics']['auc_essential_classification']}")
    print(f"\n{results['interpretation']['conclusion']}")
