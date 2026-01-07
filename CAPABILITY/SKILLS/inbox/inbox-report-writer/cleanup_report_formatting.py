#!/usr/bin/env python3
"""
Cleanup utility to fix formatting issues in INBOX reports.

Issues addressed:
1. Remove deprecated 'hashtags' field from YAML frontmatter
2. Ensure all reports follow DOCUMENT_POLICY.md format
3. Recompute content hashes after changes
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INBOX_REPORTS = PROJECT_ROOT / "INBOX" / "reports"


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_report(content: str) -> Tuple[str, str, str]:
    """
    Parse report into frontmatter, hash comment, and body.
    
    Returns:
        (frontmatter, hash_comment, body)
    """
    # Match YAML frontmatter
    fm_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not fm_match:
        raise ValueError("No YAML frontmatter found")
    
    frontmatter = fm_match.group(1)
    rest = content[fm_match.end():]
    
    # Match hash comment
    hash_match = re.match(r'(\<\!-- CONTENT_HASH: [a-f0-9]{64} --\>)\n*', rest)
    if hash_match:
        hash_comment = hash_match.group(1)
        body = rest[hash_match.end():]
    else:
        hash_comment = ""
        body = rest
    
    return frontmatter, hash_comment, body


def clean_frontmatter(frontmatter: str) -> Tuple[str, bool]:
    """
    Remove deprecated fields from frontmatter.
    
    Returns:
        (cleaned_frontmatter, modified)
    """
    modified = False
    lines = frontmatter.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for hashtags field
        if line.startswith('hashtags:'):
            # Skip this line and any following list items
            modified = True
            i += 1
            # Skip continuation lines (list items or multiline)
            while i < len(lines) and (lines[i].startswith('- ') or lines[i].startswith('  ')):
                i += 1
            continue
        
        cleaned_lines.append(line)
        i += 1
    
    return '\n'.join(cleaned_lines), modified


def rebuild_report(frontmatter: str, body: str) -> str:
    """Rebuild complete report with correct hash."""
    # Compute hash on body only
    content_hash = compute_content_hash(body)
    
    # Assemble report
    return f"---\n{frontmatter}\n---\n<!-- CONTENT_HASH: {content_hash} -->\n{body}"


def process_report_file(file_path: Path, dry_run: bool = True) -> Dict:
    """
    Process a single report file.
    
    Returns:
        Dict with status and changes
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        frontmatter, hash_comment, body = parse_report(content)
        
        # Clean frontmatter
        cleaned_fm, modified = clean_frontmatter(frontmatter)
        
        if not modified:
            return {
                "file": str(file_path.relative_to(PROJECT_ROOT)),
                "status": "ok",
                "modified": False,
            }
        
        # Rebuild with new hash
        new_content = rebuild_report(cleaned_fm, body)
        
        if not dry_run:
            file_path.write_text(new_content, encoding='utf-8')
        
        return {
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "status": "cleaned",
            "modified": True,
            "changes": ["Removed deprecated 'hashtags' field"],
        }
    
    except Exception as e:
        return {
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "status": "error",
            "error": str(e),
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up report formatting issues")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Find all markdown files in INBOX/reports
    report_files = list(INBOX_REPORTS.rglob("*.md"))
    
    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(report_files)} reports...")
    print()
    
    results = []
    for file_path in sorted(report_files):
        result = process_report_file(file_path, dry_run=args.dry_run)
        results.append(result)
        
        if args.verbose or result["modified"]:
            status_icon = "✓" if result["status"] == "ok" else ("✎" if result["status"] == "cleaned" else "✗")
            print(f"{status_icon} {result['file']}")
            if result.get("changes"):
                for change in result["changes"]:
                    print(f"    - {change}")
            if result.get("error"):
                print(f"    ERROR: {result['error']}")
    
    # Summary
    print()
    print("=" * 60)
    total = len(results)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    cleaned_count = sum(1 for r in results if r["status"] == "cleaned")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"Total reports: {total}")
    print(f"  Already clean: {ok_count}")
    print(f"  {'Would be cleaned' if args.dry_run else 'Cleaned'}: {cleaned_count}")
    print(f"  Errors: {error_count}")
    
    if args.dry_run and cleaned_count > 0:
        print()
        print("Run without --dry-run to apply changes.")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
