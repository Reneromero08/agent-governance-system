#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download HistWords - Stanford's Historical Word Embeddings

HistWords provides pre-trained word embeddings by decade from 1800-1990,
allowing us to track actual semantic drift over 190 years of real language change.

Data source: https://nlp.stanford.edu/projects/histwords/

REAL DATA ONLY. Download manually from Stanford's website.

Usage:
    # Download instructions (manual):
    1. Go to https://nlp.stanford.edu/projects/histwords/
    2. Download "English (All)" SGNS embeddings (~2.5GB)
       OR "English Fiction" (~500MB) for faster testing
    3. Extract to: THOUGHT/LAB/FORMULA/experiments/open_questions/q37/data/histwords_data/
    4. Run Q37 Tier 1 tests
"""

import os
import sys
import requests
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HISTWORDS_URL = "https://nlp.stanford.edu/projects/histwords/"

# Direct download URLs from Stanford SNAP
DOWNLOAD_URLS = {
    'eng-all': 'http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip',
    'eng-fiction': 'http://snap.stanford.edu/historical_embeddings/eng-fiction-all_sgns.zip',
    'coha-word': 'http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip',
}


def check_histwords_data(data_dir: str) -> bool:
    """Check if HistWords data is available."""
    vocab_paths = [
        os.path.join(data_dir, 'vocab.pkl'),
        os.path.join(data_dir, 'eng-all_sgns', 'vocab.pkl'),
        os.path.join(data_dir, 'eng-fiction-all_sgns', 'vocab.pkl'),
    ]

    for vp in vocab_paths:
        if os.path.exists(vp):
            print(f"Found HistWords vocabulary at: {vp}")
            return True

    return False


def download_histwords(output_dir: str, dataset: str = 'eng-fiction') -> bool:
    """
    Attempt to download HistWords data.

    NOTE: Stanford may require manual download. If this fails,
    download manually from the website.
    """
    import zipfile

    url = DOWNLOAD_URLS.get(dataset)
    if not url:
        print(f"Unknown dataset: {dataset}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f'{dataset}_sgns.zip')

    print(f"Attempting to download HistWords ({dataset})...")
    print(f"URL: {url}")
    print(f"This may take several minutes (~500MB-2.5GB)...")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded // (1024 * 1024)
                        print(f"\r  Downloaded: {mb}MB ({pct:.1f}%)", end='', flush=True)

        print()

        # Extract
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)

        # Clean up
        os.remove(zip_path)
        print(f"HistWords data extracted to: {output_dir}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)
        print(f"1. Go to: {HISTWORDS_URL}")
        print(f"2. Download: {dataset}_sgns.zip")
        print(f"3. Extract to: {output_dir}")
        print("=" * 60)
        return False


def print_download_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 70)
    print("HISTWORDS DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\nHistWords provides historical word embeddings from 1800-1990.")
    print("This is REAL DATA required for Q37 Tier 1 tests.")
    print("\nDownload options:")
    print("\n1. English Fiction (recommended, ~500MB):")
    print(f"   {DOWNLOAD_URLS['eng-fiction']}")
    print("\n2. English All (complete, ~2.5GB):")
    print(f"   {DOWNLOAD_URLS['eng-all']}")
    print("\nExtract to:")
    print("   THOUGHT/LAB/FORMULA/experiments/open_questions/q37/data/histwords_data/")
    print("\nExpected structure after extraction:")
    print("   histwords_data/")
    print("     eng-fiction-all_sgns/ (or eng-all_sgns/)")
    print("       vocab.pkl")
    print("       1800-w.npy")
    print("       1810-w.npy")
    print("       ...")
    print("       1990-w.npy")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download HistWords historical embeddings')
    parser.add_argument('--output-dir', type=str, default='histwords_data',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, default='eng-fiction',
                        choices=['eng-fiction', 'eng-all'],
                        help='Dataset to download')
    parser.add_argument('--check', action='store_true',
                        help='Just check if data exists')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)

    if args.check:
        if check_histwords_data(output_dir):
            print("HistWords data is available!")
            sys.exit(0)
        else:
            print("HistWords data NOT found.")
            print_download_instructions()
            sys.exit(1)
    else:
        if check_histwords_data(output_dir):
            print("HistWords data already exists.")
            sys.exit(0)

        success = download_histwords(output_dir, args.dataset)
        if not success:
            print_download_instructions()
            sys.exit(1)
