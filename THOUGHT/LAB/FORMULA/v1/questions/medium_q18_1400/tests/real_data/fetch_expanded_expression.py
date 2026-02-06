#!/usr/bin/env python3
"""
Fetch EXPANDED gene expression data from multiple sources.

This script tries multiple GEO datasets and combines them for a larger sample.
Also attempts to get gene symbols instead of just probe IDs.
"""

import urllib.request
import urllib.parse
import gzip
import json
import io
import os
import ssl
from pathlib import Path
import math
import re

ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, timeout: int = 120) -> bytes:
    """Fetch URL with error handling."""
    print(f"  Fetching: {url[:80]}...")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Q18 Research Bot)',
            'Accept': '*/*'
        })
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        print(f"  Error: {e}")
        return None


def parse_series_matrix(content: str, gse_id: str, limit: int = 1000):
    """Parse GEO Series Matrix format."""
    lines = content.split('\n')

    in_table = False
    headers = []
    gene_data = {}
    count = 0

    for line in lines:
        if line.startswith('!series_matrix_table_begin'):
            in_table = True
            continue
        elif line.startswith('!series_matrix_table_end'):
            break
        elif in_table:
            if not headers:
                headers = line.strip().split('\t')
                print(f"  {gse_id}: {len(headers) - 1} samples")
            else:
                if count >= limit:
                    break

                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    probe_id = parts[0].strip('"')

                    try:
                        values = []
                        for v in parts[1:]:
                            v = v.strip('"')
                            if v and v != 'null' and v != 'NA' and v != '':
                                try:
                                    val = float(v)
                                    if not math.isnan(val) and not math.isinf(val):
                                        values.append(val)
                                except ValueError:
                                    pass

                        if len(values) >= 10:  # Require at least 10 samples
                            gene_data[probe_id] = values
                            count += 1
                    except Exception:
                        pass

    return gene_data, len(headers) - 1 if headers else 0


def fetch_multiple_datasets():
    """Fetch expression data from multiple GEO datasets."""

    # List of datasets with processed expression matrices
    datasets = [
        # Format: (GSE_ID, description)
        ("GSE13904", "Pediatric sepsis blood samples - 227 samples"),
        ("GSE3526", "Human tissue atlas - 353 samples"),
        ("GSE5847", "Breast cancer expression - 95 samples"),
        ("GSE2109", "Human tumor expression - 2158 samples"),
        ("GSE9782", "Muscle expression - 100 samples"),
        ("GSE12417", "Acute myeloid leukemia - 163 samples"),
    ]

    all_gene_data = {}
    all_sources = []
    total_samples = 0

    print("=" * 60)
    print("FETCHING MULTIPLE GEO DATASETS")
    print("=" * 60)

    for gse_id, description in datasets:
        print(f"\n{gse_id}: {description}")

        # Try series matrix
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"

        data = fetch_url(url, timeout=180)
        if data:
            try:
                content = gzip.decompress(data).decode('utf-8', errors='replace')
                gene_data, n_samples = parse_series_matrix(content, gse_id, limit=500)

                if gene_data:
                    print(f"  Got {len(gene_data)} genes from {n_samples} samples")

                    # Add prefix to distinguish datasets
                    for probe, values in gene_data.items():
                        key = f"{gse_id}:{probe}"
                        all_gene_data[key] = values

                    all_sources.append({
                        'id': gse_id,
                        'description': description,
                        'n_genes': len(gene_data),
                        'n_samples': n_samples
                    })
                    total_samples = max(total_samples, n_samples)

            except Exception as e:
                print(f"  Parse error: {e}")

    return all_gene_data, all_sources, total_samples


def compute_r_values(gene_data: dict) -> dict:
    """Compute R = mean/std for each gene."""

    genes_with_r = {}

    for gene_id, values in gene_data.items():
        if len(values) >= 10:
            mean_expr = sum(values) / len(values)
            variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
            std_expr = math.sqrt(variance) if variance > 0 else 0.0001

            R = mean_expr / std_expr if std_expr > 0.0001 else float('inf')

            if 0 < R < 1000:
                genes_with_r[gene_id] = {
                    'mean_expr': round(mean_expr, 4),
                    'std_expr': round(std_expr, 4),
                    'R': round(R, 4),
                    'n_values': len(values)
                }

    return genes_with_r


def compute_r_statistics(genes_with_r: dict) -> dict:
    """Compute summary statistics for R values."""

    r_values = [g['R'] for g in genes_with_r.values()]

    if not r_values:
        return {}

    mean_r = sum(r_values) / len(r_values)
    variance = sum((x - mean_r) ** 2 for x in r_values) / len(r_values)
    std_r = math.sqrt(variance)

    sorted_r = sorted(r_values)
    median_r = sorted_r[len(sorted_r) // 2]

    # Compute percentiles
    p25_idx = len(sorted_r) // 4
    p75_idx = 3 * len(sorted_r) // 4

    return {
        'mean_R': round(mean_r, 4),
        'std_R': round(std_r, 4),
        'median_R': round(median_r, 4),
        'min_R': round(min(r_values), 4),
        'max_R': round(max(r_values), 4),
        'p25_R': round(sorted_r[p25_idx], 4),
        'p75_R': round(sorted_r[p75_idx], 4),
        'n_genes': len(r_values)
    }


def try_ensembl_expression():
    """
    Try Ensembl Expression Atlas API.
    """
    print("\n" + "=" * 60)
    print("FETCHING ENSEMBL EXPRESSION ATLAS DATA")
    print("=" * 60)

    # Try Expression Atlas experiment data
    # E-MTAB-513 is a baseline expression experiment

    experiments = [
        "E-MTAB-513",   # Human tissues
        "E-MTAB-2836",  # Human cell lines
    ]

    for exp_id in experiments:
        print(f"\nTrying {exp_id}...")

        # Analytics API endpoint for downloading data
        url = f"https://www.ebi.ac.uk/gxa/experiments-content/{exp_id}/resources/ExperimentDownloadSupplier.RnaSeqBaseline/tpms.tsv"

        data = fetch_url(url, timeout=60)
        if data:
            try:
                content = data.decode('utf-8')
                lines = content.strip().split('\n')
                print(f"  Got {len(lines)} lines")

                if len(lines) > 10:
                    # Parse TSV
                    gene_data = {}
                    headers = None

                    for i, line in enumerate(lines):
                        if line.startswith('Gene ID') or line.startswith('Gene Name'):
                            headers = line.split('\t')
                            print(f"  Headers: {len(headers)} columns")
                        elif headers and i < 1000:
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                gene_id = parts[0]
                                try:
                                    values = [float(v) for v in parts[2:] if v and v != 'NA']
                                    if len(values) >= 3:
                                        gene_data[gene_id] = values
                                except ValueError:
                                    pass

                    if gene_data:
                        print(f"  Parsed {len(gene_data)} genes")
                        return gene_data, exp_id

            except Exception as e:
                print(f"  Parse error: {e}")

    return None, None


def main():
    print("=" * 60)
    print("Q18 EXPANDED GENE EXPRESSION FETCHER")
    print("Goal: Get REAL expression data from multiple sources")
    print("=" * 60)

    # Fetch from multiple GEO datasets
    gene_data, sources, max_samples = fetch_multiple_datasets()

    if not gene_data:
        print("\nFailed to fetch any GEO data")
        return False

    # Compute R values
    genes_with_r = compute_r_values(gene_data)
    r_stats = compute_r_statistics(genes_with_r)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total genes: {len(genes_with_r)}")
    print(f"Sources: {len(sources)}")
    for src in sources:
        print(f"  - {src['id']}: {src['n_genes']} genes, {src['n_samples']} samples")

    print(f"\nR Statistics:")
    print(f"  Mean R:   {r_stats['mean_R']:.4f}")
    print(f"  Median R: {r_stats['median_R']:.4f}")
    print(f"  Std R:    {r_stats['std_R']:.4f}")
    print(f"  Min R:    {r_stats['min_R']:.4f}")
    print(f"  Max R:    {r_stats['max_R']:.4f}")
    print(f"  P25 R:    {r_stats['p25_R']:.4f}")
    print(f"  P75 R:    {r_stats['p75_R']:.4f}")

    # Save results
    output = {
        'genes': genes_with_r,
        'n_genes': len(genes_with_r),
        'n_samples': max_samples,
        'sources': sources,
        'source': 'Multiple GEO datasets',
        'format': 'Series Matrix',
        'r_statistics': r_stats
    }

    output_file = CACHE_DIR / 'gene_expression_expanded.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Also try Ensembl for comparison
    ensembl_data, exp_id = try_ensembl_expression()
    if ensembl_data:
        ensembl_genes = compute_r_values(ensembl_data)
        ensembl_stats = compute_r_statistics(ensembl_genes)

        ensembl_output = {
            'genes': ensembl_genes,
            'n_genes': len(ensembl_genes),
            'source': f'Ensembl Expression Atlas {exp_id}',
            'r_statistics': ensembl_stats
        }

        ensembl_file = CACHE_DIR / 'ensembl_expression.json'
        with open(ensembl_file, 'w') as f:
            json.dump(ensembl_output, f, indent=2)

        print(f"\nEnsembl data saved to: {ensembl_file}")
        print(f"  Mean R: {ensembl_stats.get('mean_R', 'N/A')}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
