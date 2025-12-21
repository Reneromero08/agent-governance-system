#!/usr/bin/env python3
"""
Context Query Tool - Search and retrieve decision context.

This tool enables agents and humans to query the CONTEXT folder for:
- Decisions (ADRs) by tag, status, or content
- Preferences (STYLE records)
- Rejected approaches
- Open questions

Usage:
  python CONTEXT/query-context.py "button colors"
  python CONTEXT/query-context.py --tag architecture --status accepted
  python CONTEXT/query-context.py --review-due 30
  python CONTEXT/query-context.py --list
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = PROJECT_ROOT / "CONTEXT"


@dataclass
class ContextRecord:
    """Represents a single context record (ADR, STYLE, REJECT, OPEN)."""
    id: str
    record_type: str  # decisions, preferences, rejected, open
    title: str
    status: Optional[str] = None
    date: Optional[datetime] = None
    review: Optional[datetime] = None
    confidence: Optional[str] = None
    impact: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    path: Path = None
    content: str = ""


def parse_metadata(content: str) -> Dict[str, str]:
    """Extract metadata from markdown frontmatter-style headers."""
    metadata = {}
    for line in content.split('\n')[:30]:  # Check first 30 lines
        # Match patterns like "**Status:** Accepted" or "**Date:** 2025-12-21"
        if match := re.match(r'\*\*(\w+):\*\*\s*(.+)', line):
            key, value = match.groups()
            metadata[key.lower()] = value.strip()
        # Also match "Status: Accepted" without bold
        elif match := re.match(r'^(\w+):\s+(.+)$', line):
            key, value = match.groups()
            if key.lower() in ('status', 'date', 'review', 'confidence', 'impact', 'tags', 'type'):
                metadata[key.lower()] = value.strip()
    return metadata


def parse_record(path: Path) -> Optional[ContextRecord]:
    """Parse a context record file into a ContextRecord object."""
    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None

    metadata = parse_metadata(content)

    # Extract title from first heading
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    # Parse tags
    tags = []
    if 'tags' in metadata:
        # Handle both [tag1, tag2] and tag1, tag2 formats
        tag_str = metadata['tags'].strip('[]')
        tags = [t.strip() for t in tag_str.split(',') if t.strip()]

    # Parse dates
    date = None
    if 'date' in metadata:
        try:
            date = datetime.strptime(metadata['date'].strip(), '%Y-%m-%d')
        except ValueError:
            pass

    review = None
    if 'review' in metadata:
        try:
            review = datetime.strptime(metadata['review'].strip(), '%Y-%m-%d')
        except ValueError:
            pass

    # Determine record type from parent directory
    record_type = path.parent.name

    return ContextRecord(
        id=path.stem,
        record_type=record_type,
        title=title,
        status=metadata.get('status'),
        date=date,
        review=review,
        confidence=metadata.get('confidence'),
        impact=metadata.get('impact'),
        tags=tags,
        path=path,
        content=content
    )


def load_all_records() -> List[ContextRecord]:
    """Load all context records from the CONTEXT directory."""
    records = []
    subdirs = ['decisions', 'preferences', 'rejected', 'open']

    for subdir in subdirs:
        dir_path = CONTEXT_DIR / subdir
        if not dir_path.exists():
            continue

        for md_file in dir_path.glob('*.md'):
            # Skip templates
            if 'template' in md_file.stem.lower() or md_file.stem.startswith('_'):
                continue
            if record := parse_record(md_file):
                records.append(record)

    return records


def search_records(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
    record_type: Optional[str] = None,
    review_due_days: Optional[int] = None
) -> List[ContextRecord]:
    """Search context records with filters."""
    records = load_all_records()
    filtered = records

    # Text search in title and content
    if query:
        query_lower = query.lower()
        filtered = [r for r in filtered if
                   query_lower in r.title.lower() or
                   query_lower in r.content.lower()]

    # Filter by tags
    if tags:
        filtered = [r for r in filtered if
                   any(tag.lower() in [t.lower() for t in r.tags] for tag in tags)]

    # Filter by status
    if status:
        filtered = [r for r in filtered if
                   r.status and status.lower() in r.status.lower()]

    # Filter by record type
    if record_type:
        filtered = [r for r in filtered if
                   r.record_type == record_type]

    # Filter by review due date
    if review_due_days is not None:
        cutoff = datetime.now() + timedelta(days=review_due_days)
        filtered = [r for r in filtered if
                   r.review and r.review <= cutoff]

    return filtered


def format_record(record: ContextRecord, verbose: bool = False) -> str:
    """Format a single record for display."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"{record.record_type.upper()}: {record.title}")
    lines.append(f"ID: {record.id}")

    if record.status:
        lines.append(f"Status: {record.status}")
    if record.date:
        lines.append(f"Date: {record.date.strftime('%Y-%m-%d')}")
    if record.review:
        lines.append(f"Review: {record.review.strftime('%Y-%m-%d')}")
    if record.confidence:
        lines.append(f"Confidence: {record.confidence}")
    if record.impact:
        lines.append(f"Impact: {record.impact}")
    if record.tags:
        lines.append(f"Tags: {', '.join(record.tags)}")

    lines.append(f"Path: {record.path.relative_to(PROJECT_ROOT)}")

    if verbose:
        lines.append("")
        lines.append(record.content)
    else:
        # Show first paragraph after title
        paragraphs = record.content.split('\n\n')
        if len(paragraphs) > 1:
            preview = paragraphs[1][:200]
            if len(paragraphs[1]) > 200:
                preview += "..."
            lines.append(f"\n{preview}")

    lines.append("")
    return '\n'.join(lines)


def format_json(records: List[ContextRecord]) -> str:
    """Format records as JSON for programmatic use."""
    output = []
    for r in records:
        output.append({
            'id': r.id,
            'type': r.record_type,
            'title': r.title,
            'status': r.status,
            'date': r.date.isoformat() if r.date else None,
            'review': r.review.isoformat() if r.review else None,
            'confidence': r.confidence,
            'impact': r.impact,
            'tags': r.tags,
            'path': str(r.path.relative_to(PROJECT_ROOT))
        })
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Query AGS context records (ADRs, preferences, rejected, open)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CONTEXT/query-context.py "authentication"
  python CONTEXT/query-context.py --tag architecture
  python CONTEXT/query-context.py --status accepted
  python CONTEXT/query-context.py --type decisions
  python CONTEXT/query-context.py --review-due 30
  python CONTEXT/query-context.py --list --json
        """
    )
    parser.add_argument('query', nargs='?', help="Text to search in titles and content")
    parser.add_argument('--tag', action='append', dest='tags', help="Filter by tag (can repeat)")
    parser.add_argument('--status', help="Filter by status (accepted, experimental, deprecated)")
    parser.add_argument('--type', dest='record_type', choices=['decisions', 'preferences', 'rejected', 'open'],
                       help="Filter by record type")
    parser.add_argument('--review-due', type=int, metavar='DAYS',
                       help="Show records due for review within N days")
    parser.add_argument('--list', action='store_true', help="List all records (summary)")
    parser.add_argument('--json', action='store_true', help="Output as JSON")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show full content")

    args = parser.parse_args()

    # If no filters provided and not listing, show help
    if not any([args.query, args.tags, args.status, args.record_type, args.review_due, args.list]):
        parser.print_help()
        return 0

    results = search_records(
        query=args.query,
        tags=args.tags,
        status=args.status,
        record_type=args.record_type,
        review_due_days=args.review_due
    )

    if args.list and not any([args.query, args.tags, args.status, args.record_type, args.review_due]):
        # List all records
        results = load_all_records()

    if not results:
        print("No records found.")
        return 0

    if args.json:
        print(format_json(results))
    else:
        print(f"\nFound {len(results)} record(s):\n")
        for record in results:
            print(format_record(record, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
