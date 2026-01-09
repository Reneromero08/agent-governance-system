#!/usr/bin/env python3
"""
Fix AGS skills to have Claude Code-compatible YAML frontmatter.

Transforms:
    <!-- CONTENT_HASH: ... -->
    **required_canon_version:** >=3.0.0
    # Skill: skill-name
    **Version:** 1.0.0
    ## Purpose
    Description here...

Into:
    ---
    name: skill-name
    description: Description here (first sentence/paragraph)
    ---
    <!-- CONTENT_HASH: ... -->
    ...rest of content...
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO_ROOT / "CAPABILITY" / "SKILLS"


def extract_skill_info(content: str) -> dict:
    """Extract skill name and description from AGS SKILL.md format."""
    info = {
        "name": None,
        "description": None,
        "has_frontmatter": content.strip().startswith("---"),
    }

    # Already has YAML frontmatter
    if info["has_frontmatter"]:
        return info

    # Extract skill name from "# Skill: name"
    name_match = re.search(r'^#\s*Skill:\s*(.+)$', content, re.MULTILINE)
    if name_match:
        info["name"] = name_match.group(1).strip().lower()

    # Try multiple patterns for description
    description = None

    # Pattern 1: "## Purpose" section
    purpose_match = re.search(
        r'^##\s*Purpose\s*\n+(.+?)(?=\n##|\n\*\*|\Z)',
        content,
        re.MULTILINE | re.DOTALL
    )
    if purpose_match:
        description = purpose_match.group(1).strip()

    # Pattern 2: "## Trigger" section (for simpler skills)
    if not description:
        trigger_match = re.search(
            r'^##\s*Trigger\s*\n+(.+?)(?=\n##|\n\*\*|\Z)',
            content,
            re.MULTILINE | re.DOTALL
        )
        if trigger_match:
            description = trigger_match.group(1).strip()

    # Pattern 3: Second level-1 header with same name (e.g., "# Ant Worker")
    if not description and info["name"]:
        # Look for a line after "# Skill Name" that's a description
        second_h1_match = re.search(
            r'^#\s+' + re.escape(info["name"].replace("-", " ").title()) + r'\s*\n+(.+?)(?=\n##|\n```|\Z)',
            content,
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        if second_h1_match:
            description = second_h1_match.group(1).strip()

    # Pattern 4: First non-metadata paragraph after Status line
    if not description:
        status_match = re.search(
            r'\*\*Status:\*\*.*?\n\n+(.+?)(?=\n##|\n\*\*|\n```|\Z)',
            content,
            re.DOTALL
        )
        if status_match:
            description = status_match.group(1).strip()

    if description:
        # Get first paragraph or first sentence
        first_para = description.split('\n\n')[0].strip()
        # Remove markdown formatting
        first_para = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', first_para)  # Links
        first_para = re.sub(r'`([^`]+)`', r'\1', first_para)  # Code
        first_para = re.sub(r'\*\*([^*]+)\*\*', r'\1', first_para)  # Bold
        first_para = re.sub(r'\s+', ' ', first_para)  # Whitespace
        # Truncate if too long (keep under 200 chars)
        if len(first_para) > 200:
            first_para = first_para[:197] + "..."
        info["description"] = first_para

    return info


def add_frontmatter(content: str, name: str, description: str) -> str:
    """Add YAML frontmatter to skill content."""
    # Escape any quotes in description
    description = description.replace('"', '\\"')

    frontmatter = f'''---
name: {name}
description: "{description}"
---
'''
    return frontmatter + content


def process_skill(skill_path: Path, dry_run: bool = False) -> dict:
    """Process a single SKILL.md file."""
    content = skill_path.read_text(encoding='utf-8')
    info = extract_skill_info(content)

    result = {
        "path": str(skill_path.relative_to(REPO_ROOT)),
        "name": info["name"],
        "description": info["description"],
        "status": "skipped",
        "reason": None,
    }

    if info["has_frontmatter"]:
        result["status"] = "skipped"
        result["reason"] = "already has frontmatter"
        return result

    if not info["name"]:
        result["status"] = "error"
        result["reason"] = "could not extract skill name"
        return result

    if not info["description"]:
        result["status"] = "error"
        result["reason"] = "could not extract description"
        return result

    # Add frontmatter
    new_content = add_frontmatter(content, info["name"], info["description"])

    if dry_run:
        result["status"] = "would_update"
        result["preview"] = new_content[:500]
    else:
        skill_path.write_text(new_content, encoding='utf-8')
        result["status"] = "updated"

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix AGS skill frontmatter for Claude Code")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--skill", type=str, help="Process only this skill (by name)")
    args = parser.parse_args()

    # Find all SKILL.md files
    skill_files = list(SKILLS_ROOT.glob("**/SKILL.md"))
    skill_files = [f for f in skill_files if "_TEMPLATE" not in str(f)]

    print(f"Found {len(skill_files)} skills")
    print("=" * 60)

    results = {"updated": 0, "skipped": 0, "error": 0}

    for skill_path in sorted(skill_files):
        if args.skill:
            if args.skill not in str(skill_path):
                continue

        result = process_skill(skill_path, dry_run=args.dry_run)
        status = result["status"]

        if status == "updated" or status == "would_update":
            results["updated"] += 1
            icon = "[+]"
        elif status == "skipped":
            results["skipped"] += 1
            icon = "[=]"
        else:
            results["error"] += 1
            icon = "[!]"

        print(f"{icon} {result['path']}")
        print(f"    name: {result['name']}")
        if result.get("description"):
            desc_preview = result["description"][:60] + "..." if len(result.get("description", "")) > 60 else result.get("description", "")
            print(f"    desc: {desc_preview}")
        if result.get("reason"):
            print(f"    reason: {result['reason']}")
        print()

    print("=" * 60)
    print(f"Updated: {results['updated']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Errors:  {results['error']}")

    if args.dry_run:
        print("\n(dry run - no files modified)")


if __name__ == "__main__":
    main()
