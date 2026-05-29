#!/usr/bin/env python3
"""
CONSTELLATION ARCHITECT - GOD TIER Paper Transformer
Normalizes markdown headers to proper ## ### #### ##### hierarchy.

Two modes:
1. REGEX mode (default): Parse existing markdown headers, normalize levels
2. LLM mode: Use Nemotron for papers without clear headers

Nemotron endpoint: http://10.5.0.2:1234 (OpenAI-compatible)
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
import requests

# Configuration
NEMOTRON_URL = "http://10.5.0.2:1234/v1/chat/completions"
MODEL = "nemotron-3-nano-30b-a3b"
MIN_HEADERS_THRESHOLD = 5  # Papers need at least this many headers


def extract_headers_regex(content: str) -> list[dict]:
    """
    Extract headers from markdown using regex.
    Returns list of {"text": str, "level": int, "line_num": int}
    """
    headers = []
    lines = content.split('\n')

    for i, line in enumerate(lines):
        # Match markdown headers: # Header, ## Header, etc.
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            hashes = match.group(1)
            text = match.group(2).strip()

            # Skip image/figure captions, TOC markers, empty headers
            if text.startswith('!') or text.startswith('[') or not text:
                continue
            # Skip single-word headers that look like labels
            if len(text.split()) == 1 and len(text) < 3:
                continue

            headers.append({
                "text": text,
                "level": len(hashes),
                "line_num": i
            })

    return headers


def normalize_header_levels(headers: list[dict]) -> list[dict]:
    """
    Normalize header levels so main sections start at level 2.
    - If min level is 1, shift all up by 1 (# -> ##)
    - If headers use 4+ for inline labels like "Encoder:", keep them
    """
    if not headers:
        return headers

    min_level = min(h["level"] for h in headers)

    # If already starting at 2+, no shift needed
    if min_level >= 2:
        return headers

    # Shift everything up by (2 - min_level)
    shift = 2 - min_level

    normalized = []
    for h in headers:
        new_level = min(h["level"] + shift, 5)  # Cap at #####
        normalized.append({
            "text": h["text"],
            "level": new_level,
            "line_num": h["line_num"]
        })

    return normalized


def reconstruct_god_tier(content: str, headers: list[dict]) -> str:
    """
    Reconstruct paper with normalized header hierarchy.
    """
    lines = content.split('\n')
    output = []

    # Create line_num -> normalized header mapping
    header_map = {h["line_num"]: h for h in headers}

    for i, line in enumerate(lines):
        if i in header_map:
            h = header_map[i]
            # Replace with normalized header
            output.append("#" * h["level"] + " " + h["text"])
        else:
            # Keep line as-is, but clean up malformed headers
            stripped = line.strip()
            if stripped.startswith('#') and not re.match(r'^#{1,6}\s+\S', stripped):
                # Malformed header (no space or no text) - convert to paragraph
                output.append(stripped.lstrip('#').strip())
            else:
                output.append(line)

    # Clean up excessive blank lines
    result = '\n'.join(output)
    result = re.sub(r'\n{4,}', '\n\n\n', result)

    return result


def process_paper_regex(input_path: Path, output_path: Path) -> bool:
    """Process paper using regex-based header extraction and normalization."""
    try:
        content = input_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  [ERROR] Read failed: {e}")
        return False

    # Extract existing headers
    headers = extract_headers_regex(content)

    if len(headers) < MIN_HEADERS_THRESHOLD:
        print(f"  [WARN] Only {len(headers)} headers found - may need LLM processing")
        # Still proceed with what we have

    # Normalize levels
    normalized = normalize_header_levels(headers)

    # Reconstruct
    god_tier = reconstruct_god_tier(content, normalized)

    # Write output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(god_tier, encoding='utf-8')
        return True
    except Exception as e:
        print(f"  [ERROR] Write failed: {e}")
        return False


def test_regex_mode(test_papers: list[Path], output_dir: Path):
    """Test regex mode on a few papers."""
    print("\n" + "="*70)
    print("CONSTELLATION ARCHITECT - REGEX MODE TEST")
    print("="*70)
    print(f"Testing on {len(test_papers)} papers...")

    for paper_path in test_papers:
        print(f"\nProcessing: {paper_path.name}")

        # Read and extract headers
        content = paper_path.read_text(encoding='utf-8', errors='replace')
        headers = extract_headers_regex(content)
        normalized = normalize_header_levels(headers)

        print(f"  Found {len(headers)} headers")
        print(f"  Level distribution: {dict((l, sum(1 for h in normalized if h['level']==l)) for l in range(1,6))}")
        print(f"  Sample headers:")
        for h in normalized[:8]:
            print(f"    {'#' * h['level']} {h['text'][:50]}...")

        # Process
        output_path = output_dir / paper_path.name
        if process_paper_regex(paper_path, output_path):
            print(f"  [SAVED] {output_path}")
        else:
            print(f"  [FAILED]")

    print("\n" + "="*70)
    print("Test complete. Check output files.")
    print("="*70)


def process_all_papers(input_dir: Path, output_dir: Path):
    """Process all papers in directory."""
    papers = list(input_dir.glob("*.md"))
    print(f"\n{'='*70}")
    print(f"CONSTELLATION ARCHITECT - FULL RUN (REGEX MODE)")
    print(f"{'='*70}")
    print(f"Processing {len(papers)} papers...")

    results = {"success": [], "failed": [], "low_headers": []}

    for i, paper_path in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {paper_path.name}...", end=" ", flush=True)

        output_path = output_dir / paper_path.name

        # Extract headers first to check quality
        content = paper_path.read_text(encoding='utf-8', errors='replace')
        headers = extract_headers_regex(content)

        if process_paper_regex(paper_path, output_path):
            print(f"OK ({len(headers)} headers)")
            results["success"].append(paper_path.name)
            if len(headers) < MIN_HEADERS_THRESHOLD:
                results["low_headers"].append(paper_path.name)
        else:
            print("FAILED")
            results["failed"].append(paper_path.name)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Success: {len(results['success'])}/{len(papers)}")
    print(f"Failed:  {len(results['failed'])}/{len(papers)}")
    print(f"Low headers (may need review): {len(results['low_headers'])}")

    if results["low_headers"]:
        print(f"\nPapers with <{MIN_HEADERS_THRESHOLD} headers:")
        for name in results["low_headers"][:10]:
            print(f"  - {name}")

    return results


if __name__ == "__main__":
    input_dir = Path(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FERAL_RESIDENT\perception\research\papers\markdown")
    output_dir = Path(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FERAL_RESIDENT\perception\research\papers\god_tier")

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full run
        process_all_papers(input_dir, output_dir)
    else:
        # Test run (first 3 papers)
        test_papers = list(input_dir.glob("*.md"))[:3]
        print("Running test mode (3 papers). Use --full for all papers.")
        test_regex_mode(test_papers, output_dir)
