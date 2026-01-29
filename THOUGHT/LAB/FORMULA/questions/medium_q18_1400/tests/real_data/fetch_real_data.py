#!/usr/bin/env python3
"""
Q18 Real Data Fetcher

Downloads REAL biological data for Q18 tests.
NO synthetic data generation.

Data Sources:
1. ARCHS4 - Gene expression (human and mouse)
2. DepMap - Essentiality scores
3. AlphaFold - Protein structures
4. Ensembl - Ortholog mapping

Run:
    python fetch_real_data.py --all
    python fetch_real_data.py --source archs4
    python fetch_real_data.py --source depmap
"""

import argparse
import urllib.request
import urllib.parse
import gzip
import json
import os
import sys
from pathlib import Path
import ssl

# Disable SSL verification for some servers
ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file with progress indicator."""
    print(f"Downloading {description or url}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        # Create request with headers
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Q18 Research)',
            'Accept': '*/*'
        })

        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = response.headers.get('Content-Length')
            if total_size:
                total_size = int(total_size)
                print(f"  Size: {total_size / 1e6:.1f} MB")

            # Read in chunks
            chunk_size = 8192
            downloaded = 0

            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

            print(f"\n  Done: {output_path}")
            return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def fetch_depmap_essentiality():
    """
    Fetch DepMap CRISPR essentiality data.

    DepMap provides gene essentiality scores from CRISPR knockout screens.
    This is REAL experimental data measuring cell viability after gene knockout.
    """
    print("\n" + "=" * 60)
    print("FETCHING DEPMAP ESSENTIALITY DATA")
    print("=" * 60)

    # DepMap 23Q4 release
    # This is the gene effect file from CRISPR screens
    urls = [
        # Try different DepMap release URLs
        "https://ndownloader.figshare.com/files/43746708",  # CRISPRGeneEffect 23Q4
        "https://figshare.com/ndownloader/files/43746708",
    ]

    output_file = CACHE_DIR / 'depmap_gene_effect.csv'

    if output_file.exists():
        print(f"Already downloaded: {output_file}")
        return True

    for url in urls:
        if download_file(url, output_file, "DepMap CRISPR Gene Effect"):
            return True

    print("\nManual download instructions:")
    print("1. Go to https://depmap.org/portal/download/all/")
    print("2. Download 'CRISPRGeneEffect.csv' from the latest release")
    print(f"3. Save to: {output_file}")
    return False


def fetch_archs4_gene_expression():
    """
    Fetch ARCHS4 gene expression data.

    ARCHS4 provides uniformly processed RNA-seq from GEO.
    This is REAL human gene expression data.
    """
    print("\n" + "=" * 60)
    print("FETCHING ARCHS4 GENE EXPRESSION DATA")
    print("=" * 60)

    # ARCHS4 provides HDF5 files - these are large (several GB)
    # For testing, we'll use their API to get a smaller sample

    print("\nARCHS4 data is large (>10GB for full matrix).")
    print("Options:")
    print("  1. Use ARCHS4 API for smaller samples")
    print("  2. Download full matrix from https://maayanlab.cloud/archs4/download.html")

    # Try to get a sample via API
    api_url = "https://maayanlab.cloud/archs4/data/api"

    print("\nAttempting to fetch sample via API...")

    # Get list of available samples (smaller request)
    try:
        sample_url = "https://maayanlab.cloud/archs4/search/human_gene.json?q=TP53"
        req = urllib.request.Request(sample_url, headers={'User-Agent': 'Q18-Research'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            print(f"  API accessible, found {len(data.get('results', []))} results")
    except Exception as e:
        print(f"  API access failed: {e}")

    print("\nFor full gene expression matrix:")
    print("1. Go to https://maayanlab.cloud/archs4/download.html")
    print("2. Download 'human_gene_v2.4.h5' (human gene-level)")
    print(f"3. Save to: {CACHE_DIR / 'archs4_human.h5'}")
    print("4. Optionally download 'mouse_gene_v2.4.h5' for cross-species")

    return False


def fetch_alphafold_sample():
    """
    Fetch sample protein structures from AlphaFold.

    AlphaFold provides predicted structures with pLDDT confidence scores.
    """
    print("\n" + "=" * 60)
    print("FETCHING ALPHAFOLD SAMPLE DATA")
    print("=" * 60)

    # Sample UniProt IDs for testing
    sample_ids = [
        "P00533",  # EGFR
        "P04637",  # TP53
        "P38398",  # BRCA1
        "P42574",  # Caspase-3
        "P00519",  # ABL1
    ]

    output_dir = CACHE_DIR / 'alphafold'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(sample_ids)} sample proteins...")

    success_count = 0
    for uniprot_id in sample_ids:
        # AlphaFold API URL (v6 is current version as of 2026)
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
        output_file = output_dir / f"{uniprot_id}.pdb"

        if output_file.exists():
            print(f"  {uniprot_id}: already downloaded")
            success_count += 1
            continue

        if download_file(url, output_file, f"AlphaFold {uniprot_id}"):
            success_count += 1

    print(f"\nDownloaded {success_count}/{len(sample_ids)} proteins")

    # Extract pLDDT from PDB files
    print("\nExtracting pLDDT scores from PDB files...")
    plddt_data = {}

    for pdb_file in output_dir.glob("*.pdb"):
        uniprot_id = pdb_file.stem
        plddt_values = []

        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        # pLDDT is stored in B-factor column (columns 61-66)
                        try:
                            plddt = float(line[60:66].strip())
                            plddt_values.append(plddt)
                        except (ValueError, IndexError):
                            pass

            if plddt_values:
                plddt_data[uniprot_id] = {
                    'mean_plddt': sum(plddt_values) / len(plddt_values),
                    'n_residues': len(set(plddt_values))  # Approximate
                }
                print(f"  {uniprot_id}: mean pLDDT = {plddt_data[uniprot_id]['mean_plddt']:.1f}")
        except Exception as e:
            print(f"  {uniprot_id}: error - {e}")

    # Save pLDDT data
    plddt_file = CACHE_DIR / 'alphafold_plddt.json'
    with open(plddt_file, 'w') as f:
        json.dump(plddt_data, f, indent=2)
    print(f"\nSaved pLDDT data to: {plddt_file}")

    return success_count > 0


def fetch_uniprot_sequences():
    """
    Fetch protein sequences from UniProt.
    """
    print("\n" + "=" * 60)
    print("FETCHING UNIPROT SEQUENCES")
    print("=" * 60)

    # Same proteins as AlphaFold
    sample_ids = ["P00533", "P04637", "P38398", "P42574", "P00519"]

    output_file = CACHE_DIR / 'uniprot_sequences.json'

    if output_file.exists():
        print(f"Already downloaded: {output_file}")
        return True

    sequences = {}
    print(f"Fetching {len(sample_ids)} sequences...")

    for uniprot_id in sample_ids:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Q18-Research'})
            with urllib.request.urlopen(req, timeout=30) as response:
                fasta = response.read().decode()
                # Parse FASTA
                lines = fasta.strip().split('\n')
                header = lines[0]
                sequence = ''.join(lines[1:])
                sequences[uniprot_id] = {
                    'header': header,
                    'sequence': sequence,
                    'length': len(sequence)
                }
                print(f"  {uniprot_id}: {len(sequence)} residues")
        except Exception as e:
            print(f"  {uniprot_id}: error - {e}")

    # Save
    with open(output_file, 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"\nSaved sequences to: {output_file}")

    return len(sequences) > 0


def fetch_ortholog_mapping():
    """
    Fetch human-mouse ortholog mapping from Ensembl.
    """
    print("\n" + "=" * 60)
    print("FETCHING ORTHOLOG MAPPING")
    print("=" * 60)

    output_file = CACHE_DIR / 'human_mouse_orthologs.json'

    if output_file.exists():
        print(f"Already downloaded: {output_file}")
        return True

    # Ensembl BioMart REST API
    # This query gets human-mouse orthologs
    biomart_url = "http://www.ensembl.org/biomart/martservice"

    query = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Filter name="with_mmusculus_homolog" excluded="0"/>
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="mmusculus_homolog_ensembl_gene"/>
        <Attribute name="mmusculus_homolog_associated_gene_name"/>
        <Attribute name="mmusculus_homolog_orthology_type"/>
    </Dataset>
</Query>"""

    print("Querying Ensembl BioMart for orthologs...")
    print("(This may take a few minutes)")

    try:
        data = urllib.parse.urlencode({'query': query}).encode()
        req = urllib.request.Request(biomart_url, data=data)
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')

        with urllib.request.urlopen(req, timeout=300) as response:
            result = response.read().decode()

        # Parse TSV
        lines = result.strip().split('\n')
        header = lines[0].split('\t')
        orthologs = []

        for line in lines[1:1001]:  # First 1000 for testing
            fields = line.split('\t')
            if len(fields) >= 4:
                orthologs.append({
                    'human_gene_id': fields[0],
                    'human_gene_name': fields[1],
                    'mouse_gene_id': fields[2],
                    'mouse_gene_name': fields[3]
                })

        print(f"  Found {len(orthologs)} ortholog pairs")

        with open(output_file, 'w') as f:
            json.dump(orthologs, f, indent=2)
        print(f"  Saved to: {output_file}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://www.ensembl.org/biomart/martview")
        print("2. Select 'Ensembl Genes' -> 'Human genes'")
        print("3. Add filter: 'with Mouse ortholog'")
        print("4. Select attributes: Gene ID, Gene name, Mouse gene ID, Mouse gene name")
        print("5. Export as TSV")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fetch real biological data for Q18')
    parser.add_argument('--all', action='store_true', help='Fetch all data sources')
    parser.add_argument('--source', type=str, help='Specific source: depmap, archs4, alphafold, uniprot, orthologs')
    args = parser.parse_args()

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q18 REAL DATA FETCHER")
    print("Downloading REAL biological data - NO synthetic data")
    print("=" * 60)

    if args.all or args.source == 'depmap':
        fetch_depmap_essentiality()

    if args.all or args.source == 'archs4':
        fetch_archs4_gene_expression()

    if args.all or args.source == 'alphafold':
        fetch_alphafold_sample()

    if args.all or args.source == 'uniprot':
        fetch_uniprot_sequences()

    if args.all or args.source == 'orthologs':
        fetch_ortholog_mapping()

    if not args.all and not args.source:
        print("\nUsage:")
        print("  python fetch_real_data.py --all          # Fetch all sources")
        print("  python fetch_real_data.py --source X     # Fetch specific source")
        print("\nSources: depmap, archs4, alphafold, uniprot, orthologs")

    print("\n" + "=" * 60)
    print("DATA FETCHING COMPLETE")
    print(f"Cache directory: {CACHE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
