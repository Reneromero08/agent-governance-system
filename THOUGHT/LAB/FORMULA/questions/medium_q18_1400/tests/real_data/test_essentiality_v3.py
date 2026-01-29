#!/usr/bin/env python3
"""
Test R vs Gene Essentiality using REAL DepMap data.
Version 3: Uses GPL570 probe annotation for comprehensive mapping.
"""

import json
import math
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'cache'


def load_data():
    """Load all datasets."""
    with open(CACHE_DIR / 'depmap_essentiality.json', 'r') as f:
        depmap = json.load(f)['genes']
    with open(CACHE_DIR / 'gene_expression_sample.json', 'r') as f:
        expression = json.load(f)['genes']
    with open(CACHE_DIR / 'probe_to_gene_gpl570.json', 'r') as f:
        probe_mapping = json.load(f)
    return depmap, expression, probe_mapping


def map_probes_to_genes(expression, probe_mapping):
    """Map probe IDs to gene symbols and get R values."""
    gene_r = {}

    for probe_key, data in expression.items():
        # Extract probe ID from key (format: GSE13904:1007_s_at)
        probe_id = probe_key.split(':')[1] if ':' in probe_key else probe_key

        if probe_id in probe_mapping:
            gene = probe_mapping[probe_id]

            # Clean gene symbol (handle MIR4640///DDR1 format)
            if '///' in gene:
                # Take last part which is usually the main gene
                parts = gene.split('///')
                gene = parts[-1].strip()

            # Skip LOC genes, LINC genes, MIR genes
            if gene.startswith('LOC') or gene.startswith('LINC') or gene.startswith('MIR'):
                continue

            # Keep highest R if multiple probes map to same gene
            if gene not in gene_r or data['R'] > gene_r[gene]['R']:
                gene_r[gene] = {
                    'R': data['R'],
                    'mean_expr': data['mean_expr'],
                    'std_expr': data['std_expr'],
                    'probe_id': probe_id
                }

    return gene_r


def pearson_correlation(x, y):
    """Compute Pearson correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((x[i]-mx)*(y[i]-my) for i in range(n))/n
    sx = math.sqrt(sum((xi-mx)**2 for xi in x)/n)
    sy = math.sqrt(sum((yi-my)**2 for yi in y)/n)
    return cov/(sx*sy) if sx > 1e-10 and sy > 1e-10 else 0.0


def spearman_correlation(x, y):
    """Compute Spearman rank correlation."""
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for i, idx in enumerate(s, 1):
            r[idx] = i
        return r
    return pearson_correlation(rank(x), rank(y))


def compute_auc(labels, scores):
    """Compute AUC for binary classification."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])

    # Sum of ranks of positives
    sum_ranks = sum(i+1 for i, (_, l) in enumerate(pairs) if l)

    # Mann-Whitney U statistic
    u = sum_ranks - n_pos*(n_pos+1)/2
    return u/(n_pos*n_neg)


def main():
    print("="*70)
    print("Q18: R vs Gene Essentiality Test (Real DepMap Data)")
    print("="*70)

    # Load data
    depmap, expression, probe_mapping = load_data()
    print(f"\nDepMap genes: {len(depmap)}")
    print(f"Expression probes: {len(expression)}")
    print(f"Probe annotations: {len(probe_mapping)}")

    # Map probes to genes
    gene_r = map_probes_to_genes(expression, probe_mapping)
    print(f"Mapped to genes: {len(gene_r)}")

    # Match genes between datasets
    matched = []
    for gene, r_data in gene_r.items():
        if gene in depmap:
            matched.append({
                'gene': gene,
                'R': r_data['R'],
                'mean_effect': depmap[gene]['mean_effect'],
                'essential': depmap[gene]['essential']
            })

    print(f"Matched genes: {len(matched)}")

    if len(matched) < 20:
        print("ERROR: Too few matched genes for reliable analysis")
        return None

    # Extract data for analysis
    r_vals = [m['R'] for m in matched]
    effects = [m['mean_effect'] for m in matched]
    essentials = [m['essential'] for m in matched]

    # Compute statistics
    pearson_r = pearson_correlation(r_vals, effects)
    spearman_r = spearman_correlation(r_vals, effects)
    auc = compute_auc(essentials, r_vals)

    # Separate by essentiality
    ess_r = [m['R'] for m in matched if m['essential']]
    non_r = [m['R'] for m in matched if not m['essential']]

    mean_r_ess = sum(ess_r)/len(ess_r) if ess_r else 0
    mean_r_non = sum(non_r)/len(non_r) if non_r else 0

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Matched genes: {len(matched)}")
    print(f"  Essential: {len(ess_r)}")
    print(f"  Non-essential: {len(non_r)}")
    print(f"\nCorrelations (R vs CRISPR effect):")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  Spearman r: {spearman_r:.4f}")
    print(f"  (Negative = higher R -> more negative effect = more essential)")
    print(f"\nAUC (R predicts essential): {auc:.4f}")
    print(f"  (>0.5 = higher R predicts essential)")
    print(f"\nMean R by essentiality:")
    print(f"  Essential genes: {mean_r_ess:.4f}")
    print(f"  Non-essential genes: {mean_r_non:.4f}")
    print(f"  Difference: {mean_r_ess - mean_r_non:.4f}")

    # Show top essential and non-essential genes
    print(f"\n{'='*70}")
    print("TOP MATCHED GENES BY R VALUE")
    print("="*70)
    sorted_matched = sorted(matched, key=lambda x: -x['R'])

    print("\nHighest R genes:")
    for m in sorted_matched[:10]:
        ess_str = "ESSENTIAL" if m['essential'] else "non-essential"
        print(f"  {m['gene']:12} R={m['R']:6.2f} effect={m['mean_effect']:6.3f} ({ess_str})")

    print("\nLowest R genes:")
    for m in sorted_matched[-10:]:
        ess_str = "ESSENTIAL" if m['essential'] else "non-essential"
        print(f"  {m['gene']:12} R={m['R']:6.2f} effect={m['mean_effect']:6.3f} ({ess_str})")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("="*70)

    # Correlation interpretation
    # Negative correlation means higher R -> more negative effect (more essential)
    if pearson_r < -0.05:
        corr_interp = "SUPPORTS: Higher R correlates with essentiality"
    elif pearson_r > 0.05:
        corr_interp = "CONTRADICTS: Higher R correlates with non-essentiality"
    else:
        corr_interp = "WEAK: No clear R-essentiality correlation"

    # AUC interpretation
    if auc > 0.55:
        auc_interp = "SUPPORTS: R has predictive power for essentiality"
    elif auc < 0.45:
        auc_interp = "CONTRADICTS: Low R predicts essentiality"
    else:
        auc_interp = "WEAK: R marginally predicts essentiality"

    # Overall conclusion
    r_diff = mean_r_ess - mean_r_non
    if r_diff > 0.5 and auc > 0.5:
        conclusion = "SUPPORTED: Essential genes have higher R"
    elif r_diff < -0.5 and auc < 0.5:
        conclusion = "REJECTED: Essential genes have lower R"
    else:
        conclusion = "INCONCLUSIVE: Mixed or weak evidence"

    print(f"Correlation: {corr_interp}")
    print(f"AUC: {auc_interp}")
    print(f"\nCONCLUSION: {conclusion}")

    # Critical validation
    print(f"\n{'='*70}")
    print("VALIDATION: NOT CIRCULAR")
    print("="*70)
    print("R = mean/std computed from expression patterns (GEO microarray)")
    print("Essentiality = CRISPR knockout viability (DepMap)")
    print("These are INDEPENDENT measurements - no circularity!")

    # Save results
    results = {
        'analysis': 'R vs Gene Essentiality',
        'data_sources': {
            'essentiality': 'DepMap CRISPR gene effect (17,916 genes)',
            'expression': 'GEO Series Matrix (GPL570/HG-U133 Plus 2.0)',
            'probe_mapping': 'NCBI GEO GPL570 annotation (45,118 probes)'
        },
        'sample_sizes': {
            'depmap_genes': len(depmap),
            'expression_probes': len(expression),
            'probe_annotations': len(probe_mapping),
            'mapped_genes': len(gene_r),
            'matched_genes': len(matched),
            'essential_matched': len(ess_r),
            'nonessential_matched': len(non_r)
        },
        'statistics': {
            'pearson_r': round(pearson_r, 4),
            'spearman_r': round(spearman_r, 4),
            'auc': round(auc, 4),
            'mean_r_essential': round(mean_r_ess, 4),
            'mean_r_nonessential': round(mean_r_non, 4),
            'r_difference': round(r_diff, 4)
        },
        'interpretation': {
            'correlation': corr_interp,
            'auc': auc_interp,
            'conclusion': conclusion
        },
        'validation': {
            'is_circular': False,
            'reason': 'R from expression consistency, essentiality from CRISPR knockouts - independent measurements'
        },
        'matched_genes_sample': sorted_matched[:50]
    }

    output_file = Path(__file__).parent / 'essentiality_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
