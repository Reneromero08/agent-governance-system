#!/usr/bin/env python3
"""
Context Review Tool - Flag stale decisions and enforce review schedule.

This tool audits CONTEXT records for:
- Overdue reviews (past the review date)
- Upcoming reviews (within N days)
- Records missing review dates

Usage:
  python CONTEXT/review-context.py              # Show all overdue reviews
  python CONTEXT/review-context.py --due 30     # Show reviews due in 30 days
  python CONTEXT/review-context.py --all        # Show full audit
  python CONTEXT/review-context.py --json       # Output as JSON
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Import from the query tool
try:
    from query_context import load_all_records, ContextRecord
except ImportError:
    # Fallback: inline the necessary parts
    import re
    from dataclasses import dataclass, field

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CONTEXT_DIR = PROJECT_ROOT / "CONTEXT"

    @dataclass
    class ContextRecord:
        id: str
        record_type: str
        title: str
        status: str = None
        date: datetime = None
        review: datetime = None
        confidence: str = None
        impact: str = None
        tags: List[str] = field(default_factory=list)
        path: Path = None
        content: str = ""

    def parse_metadata(content: str) -> Dict[str, str]:
        metadata = {}
        for line in content.split('\n')[:30]:
            if match := re.match(r'\*\*(\w+):\*\*\s*(.+)', line):
                key, value = match.groups()
                metadata[key.lower()] = value.strip()
            elif match := re.match(r'^(\w+):\s+(.+)$', line):
                key, value = match.groups()
                if key.lower() in ('status', 'date', 'review', 'confidence', 'impact', 'tags', 'type'):
                    metadata[key.lower()] = value.strip()
        return metadata

    def parse_record(path: Path):
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return None

        metadata = parse_metadata(content)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem

        tags = []
        if 'tags' in metadata:
            tag_str = metadata['tags'].strip('[]')
            tags = [t.strip() for t in tag_str.split(',') if t.strip()]

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

        return ContextRecord(
            id=path.stem,
            record_type=path.parent.name,
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
        records = []
        for subdir in ['decisions', 'preferences', 'rejected', 'open']:
            dir_path = CONTEXT_DIR / subdir
            if not dir_path.exists():
                continue
            for md_file in dir_path.glob('*.md'):
                if 'template' in md_file.stem.lower() or md_file.stem.startswith('_'):
                    continue
                if record := parse_record(md_file):
                    records.append(record)
        return records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_reviews(warn_days: int = 30) -> Dict[str, List[ContextRecord]]:
    """
    Audit all context records for review status.

    Returns a dict with:
    - 'overdue': Records past their review date
    - 'upcoming': Records due for review within warn_days
    - 'no_review': Records without a review date set
    - 'healthy': Records with review dates in the future
    """
    records = load_all_records()
    now = datetime.now()
    warn_date = now + timedelta(days=warn_days)

    overdue = []
    upcoming = []
    no_review = []
    healthy = []

    for record in records:
        # Only check decisions and preferences for review dates
        if record.record_type not in ('decisions', 'preferences'):
            continue

        if not record.review:
            no_review.append(record)
        elif record.review < now:
            overdue.append(record)
        elif record.review <= warn_date:
            upcoming.append(record)
        else:
            healthy.append(record)

    return {
        'overdue': sorted(overdue, key=lambda r: r.review),
        'upcoming': sorted(upcoming, key=lambda r: r.review),
        'no_review': no_review,
        'healthy': healthy
    }


def format_record_summary(record: ContextRecord, now: datetime) -> str:
    """Format a single record for the review report."""
    lines = []
    lines.append(f"  {record.id}: {record.title}")

    if record.review:
        if record.review < now:
            days_overdue = (now - record.review).days
            lines.append(f"    Due: {record.review.strftime('%Y-%m-%d')} ({days_overdue} days overdue)")
        else:
            days_until = (record.review - now).days
            lines.append(f"    Due: {record.review.strftime('%Y-%m-%d')} (in {days_until} days)")

    lines.append(f"    Path: {record.path.relative_to(PROJECT_ROOT)}")
    return '\n'.join(lines)


def format_json_output(results: Dict[str, List[ContextRecord]]) -> str:
    """Format results as JSON."""
    output = {}
    now = datetime.now()

    for category, records in results.items():
        output[category] = []
        for r in records:
            item = {
                'id': r.id,
                'title': r.title,
                'type': r.record_type,
                'review': r.review.isoformat() if r.review else None,
                'path': str(r.path.relative_to(PROJECT_ROOT))
            }
            if r.review:
                if r.review < now:
                    item['days_overdue'] = (now - r.review).days
                else:
                    item['days_until'] = (r.review - now).days
            output[category].append(item)

    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Audit AGS context records for overdue reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CONTEXT/review-context.py              # Show overdue reviews
  python CONTEXT/review-context.py --due 30     # Include reviews due in 30 days
  python CONTEXT/review-context.py --all        # Full audit report
  python CONTEXT/review-context.py --json       # Machine-readable output
        """
    )
    parser.add_argument('--due', type=int, default=30, metavar='DAYS',
                       help="Flag reviews due within N days (default: 30)")
    parser.add_argument('--all', action='store_true',
                       help="Show full audit including healthy records")
    parser.add_argument('--json', action='store_true',
                       help="Output as JSON")

    args = parser.parse_args()

    results = check_reviews(warn_days=args.due)
    now = datetime.now()

    if args.json:
        print(format_json_output(results))
        return 0 if not results['overdue'] else 1

    # Text output
    print(f"\n{'='*60}")
    print("CONTEXT REVIEW AUDIT")
    print(f"Report generated: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    has_issues = False

    if results['overdue']:
        has_issues = True
        print(f"⚠️  OVERDUE REVIEWS ({len(results['overdue'])})\n")
        for record in results['overdue']:
            print(format_record_summary(record, now))
            print()

    if results['upcoming']:
        print(f"📅 UPCOMING REVIEWS - due within {args.due} days ({len(results['upcoming'])})\n")
        for record in results['upcoming']:
            print(format_record_summary(record, now))
            print()

    if results['no_review']:
        print(f"❓ MISSING REVIEW DATE ({len(results['no_review'])})\n")
        for record in results['no_review']:
            print(f"  {record.id}: {record.title}")
            print(f"    Path: {record.path.relative_to(PROJECT_ROOT)}")
            print()

    if args.all and results['healthy']:
        print(f"✅ HEALTHY RECORDS ({len(results['healthy'])})\n")
        for record in results['healthy']:
            print(format_record_summary(record, now))
            print()

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"  Overdue:        {len(results['overdue'])}")
    print(f"  Upcoming:       {len(results['upcoming'])}")
    print(f"  No review date: {len(results['no_review'])}")
    print(f"  Healthy:        {len(results['healthy'])}")
    print(f"{'='*60}\n")

    if not any([results['overdue'], results['upcoming'], results['no_review']]):
        print("✅ All reviews current. No action needed.\n")

    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
