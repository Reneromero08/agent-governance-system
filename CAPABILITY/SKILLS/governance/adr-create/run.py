#!/usr/bin/env python3
"""
Skill: adr-create

Create a new Architecture Decision Record with auto-numbering.

Contract-style wrapper:
  python run.py <input.json> <output.json>

Deterministic fixture support:
- If `dry_run` is true, returns what would be created without writing
- Otherwise, creates the ADR file
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    GuardedWriter = None


def _load_json(path: Path) -> Dict[str, Any]:
    """Load and validate JSON input."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("input must be a JSON object")
    return obj


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON output using GuardedWriter."""
    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        sys.exit(1)

    writer = GuardedWriter(
        PROJECT_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]
    )

    rel_path = path.resolve().relative_to(PROJECT_ROOT)
    writer.mkdir_auto(str(rel_path.parent))
    content = json.dumps(obj, indent=2, sort_keys=True)
    writer.write_auto(str(rel_path), content)


def find_next_adr_number(decisions_dir: Path) -> int:
    """Find the next available ADR number."""
    if not decisions_dir.exists():
        return 1

    # Use iterdir() instead of glob() to comply with filesystem access policy
    numbers = []
    for f in decisions_dir.iterdir():
        if f.is_file() and f.name.startswith("ADR-") and f.suffix == ".md":
            match = re.match(r"ADR-(\d+)", f.stem)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def generate_slug(title: str) -> str:
    """Generate a URL-safe slug from title."""
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    return slug[:40]  # Limit length


def generate_adr_content(
    adr_number: int,
    title: str,
    context: str,
    decision: str,
    status: str,
    date: str
) -> str:
    """Generate ADR markdown content."""
    return f"""# ADR-{adr_number:03d}: {title}

**Date:** {date}
**Status:** {status}
**Tags:**

## Context

{context if context else "[Describe the context and problem that led to this decision]"}

## Decision

{decision if decision else "[Describe the decision that was made]"}

## Consequences

[Describe the positive and negative consequences of this decision]

## Review

**Review Date:** [Set a date to revisit this decision, e.g., 6 months from now]
"""


def create_adr(
    title: str,
    context: str = "",
    decision: str = "",
    status: str = "proposed",
    dry_run: bool = False,
    mock_next_number: Optional[int] = None,
    mock_date: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new ADR with auto-numbering."""
    if not title:
        return {
            "created": False,
            "error": "'title' parameter is required"
        }

    decisions_dir = PROJECT_ROOT / "LAW" / "CONTEXT" / "decisions"

    # Allow mocking for deterministic tests
    if mock_next_number is not None:
        next_num = mock_next_number
    else:
        next_num = find_next_adr_number(decisions_dir)

    if mock_date is not None:
        date = mock_date
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    slug = generate_slug(title)
    filename = f"ADR-{next_num:03d}-{slug}.md"
    filepath = decisions_dir / filename
    # Normalize path to use forward slashes for cross-platform consistency
    rel_path = filepath.relative_to(PROJECT_ROOT).as_posix()

    content = generate_adr_content(next_num, title, context, decision, status, date)

    if dry_run:
        return {
            "created": False,
            "dry_run": True,
            "would_create": {
                "path": rel_path,
                "adr_number": next_num,
                "title": title,
                "filename": filename,
                "content_preview": content[:500] + "..." if len(content) > 500 else content
            }
        }

    # Actually create the file
    try:
        if not GuardedWriter:
            return {
                "created": False,
                "error": "GuardedWriter not available"
            }

        writer = GuardedWriter(
            PROJECT_ROOT,
            durable_roots=["LAW/CONTEXT/decisions"]
        )

        writer.mkdir_auto(str(decisions_dir.relative_to(PROJECT_ROOT)))
        writer.write_durable(rel_path, content)

        return {
            "created": True,
            "path": rel_path,
            "adr_number": next_num,
            "title": title,
            "message": f"Created {filename}. Please review and fill in the remaining sections."
        }
    except Exception as e:
        return {
            "created": False,
            "error": f"ADR creation error: {str(e)}"
        }


def main(argv: list) -> int:
    """Main entry point."""
    if len(argv) != 3:
        sys.stderr.write("Usage: run.py <input.json> <output.json>\n")
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2])

    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        inp = _load_json(input_path)
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    title = inp.get("title", "")
    context = inp.get("context", "")
    decision = inp.get("decision", "")
    status = inp.get("status", "proposed")
    dry_run = inp.get("dry_run", False)
    mock_next_number = inp.get("_mock_next_number")
    mock_date = inp.get("_mock_date")

    result = create_adr(
        title=title,
        context=context,
        decision=decision,
        status=status,
        dry_run=dry_run,
        mock_next_number=mock_next_number,
        mock_date=mock_date
    )

    # Merge input with result for traceability
    output = {**inp, **result}

    try:
        _write_json(output_path, output)
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
