#!/usr/bin/env python3
"""
Fetch Affymetrix HG-U133 Plus 2.0 probe annotation from NCBI GEO.
This provides probe ID -> gene symbol mapping.
"""

import urllib.request
import gzip
import json
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'

def fetch_annotation():
    """Fetch GPL570 (HG-U133 Plus 2.0) annotation from GEO."""
    print("Fetching GPL570 annotation from NCBI GEO...")

    # GPL570 is the HG-U133 Plus 2.0 platform
    # Try multiple URL formats
    urls = [
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570&targ=self&form=text&view=data",
    ]

    data = None
    for url in urls:
        print(f"Trying: {url[:60]}...")
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Q18 Research Bot)'
            })
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
                if url.endswith('.gz'):
                    data = gzip.decompress(raw).decode('utf-8', errors='replace')
                else:
                    data = raw.decode('utf-8', errors='replace')
                if data:
                    print(f"  Got {len(data)} bytes")
                    break
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not data:
        print("Failed to fetch annotation")
        return None

    # Return early to avoid duplicate try block
    return parse_annotation(data)


def parse_annotation(data):
    """Parse annotation data."""

    # Parse annotation
    probe_to_gene = {}
    lines = data.split('\n')
    header_idx = {}

    for line in lines:
        if line.startswith('#'):
            continue
        if line.startswith('ID'):
            # Header line
            parts = line.split('\t')
            for i, col in enumerate(parts):
                header_idx[col] = i
            continue

        if not line.strip():
            continue

        parts = line.split('\t')
        if len(parts) < 2:
            continue

        probe_id = parts[0]

        # Get gene symbol - column name varies
        gene_symbol = None
        for col_name in ['Gene Symbol', 'Gene symbol', 'GENE_SYMBOL']:
            if col_name in header_idx and header_idx[col_name] < len(parts):
                gene_symbol = parts[header_idx[col_name]].strip()
                break

        if gene_symbol and gene_symbol != '---' and gene_symbol != '':
            # Handle multiple gene symbols (take first)
            if ' /// ' in gene_symbol:
                gene_symbol = gene_symbol.split(' /// ')[0]
            probe_to_gene[probe_id] = gene_symbol

    print(f"Parsed {len(probe_to_gene)} probe-gene mappings")
    return probe_to_gene


def main():
    mapping = fetch_annotation()

    if mapping:
        output_file = CACHE_DIR / 'probe_to_gene_gpl570.json'
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved to {output_file}")

        # Print sample
        print("\nSample mappings:")
        for i, (probe, gene) in enumerate(list(mapping.items())[:20]):
            print(f"  {probe} -> {gene}")


if __name__ == '__main__':
    main()
