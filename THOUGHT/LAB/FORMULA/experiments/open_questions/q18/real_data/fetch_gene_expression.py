#!/usr/bin/env python3
"""
Fetch REAL gene expression data from GEO/ARCHS4.

Strategy:
1. Try ARCHS4 data API endpoints
2. Try GEO FTP for processed expression matrices
3. Try GEO GDS API for curated datasets

Goal: Get expression values for ~100-500 genes across ~50-100 samples
to compute R = mean/std for each gene.
"""

import urllib.request
import urllib.parse
import gzip
import json
import csv
import io
import os
import ssl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

# Disable SSL verification for some servers
ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, timeout: int = 60) -> Optional[bytes]:
    """Fetch URL with error handling."""
    print(f"  Fetching: {url}")
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


def try_geo_soft_format():
    """
    Try to get expression data from GEO SOFT format.

    GEO DataSets (GDS) have pre-computed expression matrices.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 1: GEO DataSet SOFT format")
    print("=" * 60)

    # Try several small GDS datasets
    # GDS datasets have curated, processed expression data
    gds_list = [
        ("GDS5244", "Human tissue expression - small dataset"),
        ("GDS4794", "Human cell line expression"),
        ("GDS3715", "Human gene expression - multiple tissues"),
    ]

    for gds_id, description in gds_list:
        print(f"\nTrying {gds_id}: {description}")

        # GDS SOFT file URL
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/datasets/{gds_id[:5]}nnn/{gds_id}/soft/{gds_id}.soft.gz"

        data = fetch_url(url, timeout=120)
        if data:
            try:
                content = gzip.decompress(data).decode('utf-8', errors='replace')
                result = parse_gds_soft(content, gds_id)
                if result and len(result['genes']) >= 100:
                    return result
            except Exception as e:
                print(f"  Parse error: {e}")

    return None


def parse_gds_soft(content: str, gds_id: str) -> Optional[Dict]:
    """Parse GDS SOFT format to extract gene expression matrix."""
    print(f"  Parsing SOFT format...")

    lines = content.split('\n')

    # Find the data table
    in_table = False
    headers = []
    gene_data = {}

    for line in lines:
        if line.startswith('!dataset_table_begin'):
            in_table = True
            continue
        elif line.startswith('!dataset_table_end'):
            break
        elif in_table:
            if not headers:
                # First line is headers
                headers = line.strip().split('\t')
                print(f"  Found {len(headers)} columns")
            else:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    gene_id = parts[0]
                    gene_symbol = parts[1]

                    # Expression values start at column 2
                    try:
                        values = []
                        for v in parts[2:]:
                            if v and v != 'null' and v != 'NA':
                                try:
                                    values.append(float(v))
                                except ValueError:
                                    pass

                        if len(values) >= 3:
                            gene_data[gene_symbol] = values
                    except Exception:
                        pass

    if not gene_data:
        print("  No gene data found in table")
        return None

    print(f"  Extracted {len(gene_data)} genes with expression values")

    # Get sample count from first gene
    first_gene = next(iter(gene_data.values()))
    n_samples = len(first_gene)

    # Compute R = mean/std for each gene
    genes_with_r = {}
    for gene, values in gene_data.items():
        if len(values) >= 3:
            mean_expr = sum(values) / len(values)
            variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
            std_expr = math.sqrt(variance) if variance > 0 else 0.0001

            R = mean_expr / std_expr if std_expr > 0.0001 else float('inf')

            if 0 < R < 1000:  # Filter extreme values
                genes_with_r[gene] = {
                    'mean_expr': round(mean_expr, 4),
                    'std_expr': round(std_expr, 4),
                    'R': round(R, 4),
                    'n_values': len(values)
                }

    return {
        'genes': genes_with_r,
        'n_samples': n_samples,
        'n_genes': len(genes_with_r),
        'source': f'GEO {gds_id}',
        'format': 'GDS SOFT'
    }


def try_geo_series_matrix():
    """
    Try to get expression data from GEO Series Matrix files.

    Series matrices are tab-delimited with genes as rows.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 2: GEO Series Matrix files")
    print("=" * 60)

    # Try several small GSE series with series matrix files
    gse_list = [
        ("GSE13904", "Human blood expression"),
        ("GSE3526", "Human tissue atlas"),
        ("GSE7307", "Human body index"),
    ]

    for gse_id, description in gse_list:
        print(f"\nTrying {gse_id}: {description}")

        # Series matrix URL
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"

        data = fetch_url(url, timeout=180)
        if data:
            try:
                content = gzip.decompress(data).decode('utf-8', errors='replace')
                result = parse_series_matrix(content, gse_id)
                if result and len(result['genes']) >= 50:
                    return result
            except Exception as e:
                print(f"  Parse error: {e}")

    return None


def parse_series_matrix(content: str, gse_id: str) -> Optional[Dict]:
    """Parse GEO Series Matrix format."""
    print(f"  Parsing Series Matrix format...")

    lines = content.split('\n')

    # Find the data table
    in_table = False
    headers = []
    gene_data = {}

    for line in lines:
        if line.startswith('!series_matrix_table_begin'):
            in_table = True
            continue
        elif line.startswith('!series_matrix_table_end'):
            break
        elif in_table:
            if not headers:
                # First line is headers (sample IDs)
                headers = line.strip().split('\t')
                print(f"  Found {len(headers) - 1} samples")
            else:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    probe_id = parts[0].strip('"')

                    # Expression values start at column 1
                    try:
                        values = []
                        for v in parts[1:]:
                            v = v.strip('"')
                            if v and v != 'null' and v != 'NA' and v != '':
                                try:
                                    values.append(float(v))
                                except ValueError:
                                    pass

                        if len(values) >= 3:
                            gene_data[probe_id] = values
                    except Exception:
                        pass

    if not gene_data:
        print("  No gene data found")
        return None

    print(f"  Extracted {len(gene_data)} probes with expression values")

    # Get sample count
    first_gene = next(iter(gene_data.values()))
    n_samples = len(first_gene)

    # Compute R = mean/std for each probe
    genes_with_r = {}
    for probe, values in list(gene_data.items())[:500]:  # Limit to 500 genes
        if len(values) >= 3:
            mean_expr = sum(values) / len(values)
            variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
            std_expr = math.sqrt(variance) if variance > 0 else 0.0001

            R = mean_expr / std_expr if std_expr > 0.0001 else float('inf')

            if 0 < R < 1000:
                genes_with_r[probe] = {
                    'mean_expr': round(mean_expr, 4),
                    'std_expr': round(std_expr, 4),
                    'R': round(R, 4),
                    'n_values': len(values)
                }

    return {
        'genes': genes_with_r,
        'n_samples': n_samples,
        'n_genes': len(genes_with_r),
        'source': f'GEO {gse_id}',
        'format': 'Series Matrix'
    }


def try_expression_atlas():
    """
    Try Expression Atlas API for gene expression data.

    Expression Atlas provides curated, normalized expression data.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 3: Expression Atlas API")
    print("=" * 60)

    # Expression Atlas baseline experiments
    experiments = [
        ("E-MTAB-513", "Human tissue RNA-seq"),
        ("E-MTAB-2836", "Human cell line expression"),
    ]

    for exp_id, description in experiments:
        print(f"\nTrying {exp_id}: {description}")

        # Expression Atlas API endpoint
        url = f"https://www.ebi.ac.uk/gxa/experiments/{exp_id}/Results"

        # This usually requires specific parameters - try basic approach
        data = fetch_url(url, timeout=60)
        if data:
            print(f"  Got response ({len(data)} bytes)")
            # Expression Atlas format varies, would need specific parsing

    return None


def try_gtex_api():
    """
    Try GTEx Portal API for tissue expression data.

    GTEx provides RNA-seq from human tissues.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 4: GTEx Portal API")
    print("=" * 60)

    # GTEx API for median gene expression by tissue
    # This endpoint gives median TPM across tissues

    print("\nFetching gene expression from GTEx API...")

    # Sample a few genes to test the API
    test_genes = [
        "TP53", "BRCA1", "EGFR", "MYC", "GAPDH", "ACTB",
        "CDK4", "RB1", "PTEN", "AKT1", "KRAS", "BRAF"
    ]

    gene_data = {}

    for gene in test_genes:
        # GTEx API endpoint for gene expression
        url = f"https://gtexportal.org/api/v2/expression/medianGeneExpression?gencodeId={gene}&datasetId=gtex_v8"

        data = fetch_url(url, timeout=30)
        if data:
            try:
                result = json.loads(data.decode())
                if 'data' in result and result['data']:
                    # Extract expression across tissues
                    tissues = result['data']
                    values = [t.get('median', 0) for t in tissues if t.get('median')]

                    if values:
                        mean_expr = sum(values) / len(values)
                        variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
                        std_expr = math.sqrt(variance) if variance > 0 else 0.0001
                        R = mean_expr / std_expr if std_expr > 0.0001 else 0

                        gene_data[gene] = {
                            'mean_expr': round(mean_expr, 4),
                            'std_expr': round(std_expr, 4),
                            'R': round(R, 4),
                            'n_tissues': len(values)
                        }
                        print(f"  {gene}: R = {R:.2f} ({len(values)} tissues)")
            except Exception as e:
                print(f"  {gene}: parse error - {e}")

    if gene_data:
        return {
            'genes': gene_data,
            'n_samples': len(gene_data),
            'n_genes': len(gene_data),
            'source': 'GTEx Portal API v8',
            'format': 'Median TPM across tissues'
        }

    return None


def try_ncbi_gene_expression():
    """
    Try NCBI Gene expression data via Entrez API.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 5: NCBI Gene Expression (via GEO Profiles)")
    print("=" * 60)

    # GEO Profiles provides expression across samples for individual genes
    # Using E-utilities to search

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Search for expression profiles
    genes = ["TP53", "BRCA1", "EGFR", "MYC", "GAPDH"]

    for gene in genes:
        # Search GEO Profiles
        search_url = f"{base_url}esearch.fcgi?db=geoprofiles&term={gene}[Gene Symbol]+AND+Homo+sapiens[Organism]&retmax=5&retmode=json"

        data = fetch_url(search_url, timeout=30)
        if data:
            try:
                result = json.loads(data.decode())
                count = result.get('esearchresult', {}).get('count', 0)
                print(f"  {gene}: {count} profiles found")
            except Exception as e:
                print(f"  {gene}: error - {e}")

    return None


def try_human_protein_atlas():
    """
    Try Human Protein Atlas API for expression data.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 6: Human Protein Atlas API")
    print("=" * 60)

    # HPA provides RNA expression data across tissues
    # Their API endpoint for RNA expression

    genes = [
        "ENSG00000141510",  # TP53
        "ENSG00000012048",  # BRCA1
        "ENSG00000146648",  # EGFR
        "ENSG00000136997",  # MYC
        "ENSG00000111640",  # GAPDH
        "ENSG00000075624",  # ACTB
        "ENSG00000257017",  # HSP90AA1
        "ENSG00000115977",  # AAK1
        "ENSG00000148400",  # NOTCH1
        "ENSG00000171862",  # PTEN
    ]

    gene_data = {}

    for ensembl_id in genes:
        # HPA API endpoint
        url = f"https://www.proteinatlas.org/{ensembl_id}.json"

        data = fetch_url(url, timeout=30)
        if data:
            try:
                result = json.loads(data.decode())

                gene_name = result.get('gene_info', {}).get('gene', ensembl_id)

                # Extract RNA tissue expression
                rna_data = result.get('rna_tissue', {})
                if rna_data:
                    values = []
                    for tissue, expr_info in rna_data.items():
                        if isinstance(expr_info, dict):
                            tpm = expr_info.get('nTPM', expr_info.get('TPM', 0))
                            if tpm and float(tpm) > 0:
                                values.append(float(tpm))

                    if len(values) >= 3:
                        mean_expr = sum(values) / len(values)
                        variance = sum((x - mean_expr) ** 2 for x in values) / len(values)
                        std_expr = math.sqrt(variance) if variance > 0 else 0.0001
                        R = mean_expr / std_expr if std_expr > 0.0001 else 0

                        gene_data[gene_name] = {
                            'mean_expr': round(mean_expr, 4),
                            'std_expr': round(std_expr, 4),
                            'R': round(R, 4),
                            'n_tissues': len(values),
                            'ensembl_id': ensembl_id
                        }
                        print(f"  {gene_name}: R = {R:.2f} ({len(values)} tissues)")
                else:
                    print(f"  {ensembl_id}: no RNA data")

            except json.JSONDecodeError:
                print(f"  {ensembl_id}: JSON parse error")
            except Exception as e:
                print(f"  {ensembl_id}: error - {e}")

    if gene_data:
        return {
            'genes': gene_data,
            'n_samples': max(g.get('n_tissues', 0) for g in gene_data.values()),
            'n_genes': len(gene_data),
            'source': 'Human Protein Atlas',
            'format': 'nTPM across tissues'
        }

    return None


def compute_r_statistics(data: Dict) -> Dict:
    """Compute summary statistics for R values."""
    genes = data.get('genes', {})

    if not genes:
        return data

    r_values = [g['R'] for g in genes.values() if 0 < g['R'] < float('inf')]

    if r_values:
        mean_r = sum(r_values) / len(r_values)
        variance = sum((x - mean_r) ** 2 for x in r_values) / len(r_values)
        std_r = math.sqrt(variance)

        # Sort R values
        sorted_r = sorted(r_values)
        median_r = sorted_r[len(sorted_r) // 2]

        data['r_statistics'] = {
            'mean_R': round(mean_r, 4),
            'std_R': round(std_r, 4),
            'median_R': round(median_r, 4),
            'min_R': round(min(r_values), 4),
            'max_R': round(max(r_values), 4),
            'n_valid_genes': len(r_values)
        }

    return data


def main():
    print("=" * 60)
    print("Q18 GENE EXPRESSION DATA FETCHER")
    print("Goal: Get REAL expression data to compute R = mean/std")
    print("=" * 60)

    results = {
        'strategies_tried': [],
        'successful_source': None,
        'data': None
    }

    # Strategy 1: GEO DataSet SOFT files
    data = try_geo_soft_format()
    results['strategies_tried'].append({
        'name': 'GEO DataSet SOFT',
        'success': data is not None,
        'n_genes': len(data['genes']) if data else 0
    })

    if data and len(data['genes']) >= 50:
        results['successful_source'] = 'GEO DataSet SOFT'
        results['data'] = compute_r_statistics(data)

    # Strategy 2: GEO Series Matrix
    if not results['data']:
        data = try_geo_series_matrix()
        results['strategies_tried'].append({
            'name': 'GEO Series Matrix',
            'success': data is not None,
            'n_genes': len(data['genes']) if data else 0
        })

        if data and len(data['genes']) >= 50:
            results['successful_source'] = 'GEO Series Matrix'
            results['data'] = compute_r_statistics(data)

    # Strategy 3: Expression Atlas
    if not results['data']:
        data = try_expression_atlas()
        results['strategies_tried'].append({
            'name': 'Expression Atlas',
            'success': data is not None,
            'n_genes': len(data['genes']) if data else 0
        })

        if data:
            results['successful_source'] = 'Expression Atlas'
            results['data'] = compute_r_statistics(data)

    # Strategy 4: GTEx API
    if not results['data']:
        data = try_gtex_api()
        results['strategies_tried'].append({
            'name': 'GTEx Portal API',
            'success': data is not None,
            'n_genes': len(data['genes']) if data else 0
        })

        if data:
            results['successful_source'] = 'GTEx Portal API'
            results['data'] = compute_r_statistics(data)

    # Strategy 5: NCBI Gene Expression
    if not results['data']:
        data = try_ncbi_gene_expression()
        results['strategies_tried'].append({
            'name': 'NCBI GEO Profiles',
            'success': data is not None,
            'n_genes': len(data['genes']) if data else 0
        })

    # Strategy 6: Human Protein Atlas
    if not results['data']:
        data = try_human_protein_atlas()
        results['strategies_tried'].append({
            'name': 'Human Protein Atlas',
            'success': data is not None,
            'n_genes': len(data['genes']) if data else 0
        })

        if data:
            results['successful_source'] = 'Human Protein Atlas'
            results['data'] = compute_r_statistics(data)

    # Save results
    output_file = CACHE_DIR / 'gene_expression_sample.json'

    if results['data']:
        with open(output_file, 'w') as f:
            json.dump(results['data'], f, indent=2)
        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Data saved to {output_file}")
        print(f"Source: {results['successful_source']}")
        print(f"Genes: {results['data']['n_genes']}")
        if 'r_statistics' in results['data']:
            stats = results['data']['r_statistics']
            print(f"Mean R: {stats['mean_R']:.4f}")
            print(f"Median R: {stats['median_R']:.4f}")
        print("=" * 60)
    else:
        print(f"\n{'=' * 60}")
        print("FAILED: Could not retrieve gene expression data")
        print("Strategies tried:")
        for s in results['strategies_tried']:
            status = "SUCCESS" if s['success'] else "FAILED"
            print(f"  - {s['name']}: {status}")
        print("=" * 60)

    # Save detailed results
    details_file = CACHE_DIR / 'data_fetch_results.json'
    with open(details_file, 'w') as f:
        json.dump({
            'strategies_tried': results['strategies_tried'],
            'successful_source': results['successful_source'],
            'has_data': results['data'] is not None
        }, f, indent=2)

    return results['data'] is not None


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
