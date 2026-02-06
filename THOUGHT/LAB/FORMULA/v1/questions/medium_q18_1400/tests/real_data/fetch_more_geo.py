#!/usr/bin/env python3
"""
Fetch more GEO datasets with corrected URLs.

GEO FTP structure:
- Series: ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE#/
- For series matrix: suppl/ or matrix/ subdirectory

Let's find datasets that definitely exist.
"""

import urllib.request
import gzip
import json
import math
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'


def fetch_url(url: str, timeout: int = 120):
    """Fetch URL with error handling."""
    print(f"  Fetching: {url[:80]}...")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
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
                n_samples = len(headers) - 1
                print(f"  Found {n_samples} samples")
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
                                val = float(v)
                                if not math.isnan(val) and not math.isinf(val):
                                    values.append(val)
                        if len(values) >= 10:
                            gene_data[probe_id] = values
                            count += 1
                    except (ValueError, TypeError):
                        pass

    return gene_data, len(headers) - 1 if headers else 0


def compute_r_stats(gene_data: dict) -> tuple:
    """Compute R = mean/std for each gene and summary stats."""
    genes_with_r = {}

    for gene_id, values in gene_data.items():
        if len(values) >= 10:
            mean_expr = sum(values) / len(values)
            variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
            std_expr = math.sqrt(variance) if variance > 0 else 0.0001
            R = mean_expr / std_expr if std_expr > 0.0001 else 0

            if 0.1 < R < 1000:
                genes_with_r[gene_id] = {
                    'mean_expr': round(mean_expr, 4),
                    'std_expr': round(std_expr, 4),
                    'R': round(R, 4),
                    'n_values': len(values)
                }

    r_values = [g['R'] for g in genes_with_r.values()]
    if not r_values:
        return genes_with_r, {}

    sorted_r = sorted(r_values)
    mean_r = sum(r_values) / len(r_values)
    variance = sum((x - mean_r) ** 2 for x in r_values) / len(r_values)

    stats = {
        'mean_R': round(mean_r, 4),
        'std_R': round(math.sqrt(variance), 4),
        'median_R': round(sorted_r[len(sorted_r) // 2], 4),
        'min_R': round(min(r_values), 4),
        'max_R': round(max(r_values), 4),
        'p25_R': round(sorted_r[len(sorted_r) // 4], 4),
        'p75_R': round(sorted_r[3 * len(sorted_r) // 4], 4),
        'n_genes': len(r_values)
    }

    return genes_with_r, stats


def try_geo_datasets():
    """Try various GEO datasets."""

    print("=" * 60)
    print("TRYING ADDITIONAL GEO DATASETS")
    print("=" * 60)

    # List of datasets with likely working URLs
    # These are popular, well-maintained datasets
    datasets = [
        # GSE + number, URL suffix pattern varies
        ("GSE32474", "GSE32474_series_matrix.txt.gz"),
        ("GSE14407", "GSE14407_series_matrix.txt.gz"),
        ("GSE36376", "GSE36376_series_matrix.txt.gz"),
        ("GSE26440", "GSE26440_series_matrix.txt.gz"),
        ("GSE3494", "GSE3494_series_matrix.txt.gz"),
        ("GSE1133", "GSE1133_series_matrix.txt.gz"),
        ("GSE7305", "GSE7305_series_matrix.txt.gz"),
        ("GSE5281", "GSE5281_series_matrix.txt.gz"),
    ]

    all_results = []

    for gse_id, filename in datasets:
        # Construct URL
        prefix = gse_id[:5] + "nnn"  # e.g., GSE32nnn
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/{filename}"

        print(f"\n{gse_id}:")
        data = fetch_url(url, timeout=180)

        if data:
            try:
                content = gzip.decompress(data).decode('utf-8', errors='replace')
                gene_data, n_samples = parse_series_matrix(content, gse_id, limit=500)

                if gene_data and len(gene_data) >= 50:
                    genes_with_r, stats = compute_r_stats(gene_data)

                    if genes_with_r:
                        all_results.append({
                            'gse_id': gse_id,
                            'n_samples': n_samples,
                            'n_genes': len(genes_with_r),
                            'genes': genes_with_r,
                            'r_statistics': stats
                        })

                        print(f"  SUCCESS: {len(genes_with_r)} genes, mean R = {stats['mean_R']:.4f}")

            except Exception as e:
                print(f"  Parse error: {e}")

    return all_results


def main():
    print("=" * 60)
    print("Q18 ADDITIONAL GEO DATA FETCHER")
    print("=" * 60)

    results = try_geo_datasets()

    if results:
        print(f"\n{'=' * 60}")
        print("COMBINED RESULTS")
        print("=" * 60)

        # Combine all genes
        all_genes = {}
        total_r_values = []

        for res in results:
            for gene_id, gene_info in res['genes'].items():
                key = f"{res['gse_id']}:{gene_id}"
                all_genes[key] = gene_info
                total_r_values.append(gene_info['R'])

        # Overall statistics
        sorted_r = sorted(total_r_values)
        mean_r = sum(total_r_values) / len(total_r_values)
        variance = sum((x - mean_r) ** 2 for x in total_r_values) / len(total_r_values)

        overall_stats = {
            'mean_R': round(mean_r, 4),
            'std_R': round(math.sqrt(variance), 4),
            'median_R': round(sorted_r[len(sorted_r) // 2], 4),
            'min_R': round(min(total_r_values), 4),
            'max_R': round(max(total_r_values), 4),
            'p25_R': round(sorted_r[len(sorted_r) // 4], 4),
            'p75_R': round(sorted_r[3 * len(sorted_r) // 4], 4),
            'n_genes': len(total_r_values)
        }

        print(f"Total genes: {len(all_genes)}")
        print(f"Mean R: {overall_stats['mean_R']:.4f}")
        print(f"Median R: {overall_stats['median_R']:.4f}")
        print(f"Std R: {overall_stats['std_R']:.4f}")

        # Save combined results
        output = {
            'genes': all_genes,
            'n_genes': len(all_genes),
            'sources': [{'gse_id': r['gse_id'], 'n_samples': r['n_samples'], 'n_genes': r['n_genes']} for r in results],
            'r_statistics': overall_stats
        }

        output_file = CACHE_DIR / 'geo_combined_expression.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved to: {output_file}")

    else:
        print("\nNo additional datasets successfully fetched")

    return len(results) > 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
