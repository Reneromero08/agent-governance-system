#!/usr/bin/env python3
"""
Codebook Generator

Scans CANON, SKILLS, CONTEXT, and other AGS components to generate
a stable ID registry (CANON/CODEBOOK.md).

Each entity gets a short, stable ID like @C1, @I3, @S7 that can be
used in prompts and documentation to reference specific rules,
invariants, skills, and decisions.

Usage:
    python TOOLS/codebook_build.py          # Generate codebook
    python TOOLS/codebook_build.py --json   # Output as JSON
    python TOOLS/codebook_build.py --check  # Verify codebook is current
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_contract_rules(contract_path: Path) -> List[Dict]:
    """Extract numbered rules from CONTRACT.md."""
    rules = []
    if not contract_path.exists():
        return rules
    
    content = contract_path.read_text(encoding="utf-8", errors="ignore")
    
    # Pattern for numbered rules: starts with "1. **" or "2. **" etc.
    # Extract the bold title and first line of description
    lines = content.split('\n')
    current_rule = None
    
    for line in lines:
        # Match: "1. **Text outranks code.**" or "7. **Commit ceremony.**"
        match = re.match(r'^(\d+)\.\s+\*\*([^*]+)\.\*\*\s*(.*)?', line)
        if match:
            num = int(match.group(1))
            title = match.group(2).strip()
            rest = match.group(3) or ""
            
            # Clean up the summary
            summary = title
            if rest:
                summary = f"{title}: {rest[:80]}"
            
            rules.append({
                "id": f"@C{num}",
                "type": "contract_rule",
                "number": num,
                "summary": summary[:100],
                "source": "CANON/CONTRACT.md"
            })
    
    return rules


def extract_invariants(invariants_path: Path) -> List[Dict]:
    """Extract invariants from INVARIANTS.md."""
    invariants = []
    if not invariants_path.exists():
        return invariants
    
    content = invariants_path.read_text(encoding="utf-8", errors="ignore")
    
    # Pattern: "- **[INV-001] Repository structure**"
    pattern = re.compile(r'-\s+\*\*\[INV-(\d+)\]\s+([^*]+)\*\*\s*[-–]?\s*(.*)', re.MULTILINE)
    
    for match in pattern.finditer(content):
        num = int(match.group(1))
        title = match.group(2).strip()
        description = match.group(3).strip()[:60] if match.group(3) else ""
        
        summary = title
        if description:
            summary = f"{title}: {description}"
        
        invariants.append({
            "id": f"@I{num}",
            "type": "invariant",
            "number": num,
            "summary": summary[:100],
            "source": "CANON/INVARIANTS.md"
        })
    
    return invariants



def extract_skills(skills_dir: Path) -> List[Dict]:
    """Extract skills from SKILLS directory."""
    skills = []
    if not skills_dir.exists():
        return skills
    
    skill_dirs = sorted([d for d in skills_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith('_')])
    
    for idx, skill_dir in enumerate(skill_dirs, 1):
        skill_md = skill_dir / "SKILL.md"
        summary = skill_dir.name
        
        if skill_md.exists():
            content = skill_md.read_text(encoding="utf-8", errors="ignore")
            # Extract first line after # Skill:
            title_match = re.search(r'#\s*(?:Skill:?\s*)?(.+)', content)
            if title_match:
                summary = title_match.group(1).strip()
        
        skills.append({
            "id": f"@S{idx}",
            "type": "skill",
            "number": idx,
            "name": skill_dir.name,
            "summary": summary[:100],
            "source": f"SKILLS/{skill_dir.name}/"
        })
    
    return skills


def extract_decisions(decisions_dir: Path) -> List[Dict]:
    """Extract ADRs from CONTEXT/decisions."""
    decisions = []
    if not decisions_dir.exists():
        return decisions
    
    adr_files = sorted(decisions_dir.glob("ADR-*.md"))
    
    for adr_file in adr_files:
        # Extract ADR number from filename
        match = re.match(r'ADR-(\d+)', adr_file.stem)
        if not match:
            continue
        
        num = int(match.group(1))
        content = adr_file.read_text(encoding="utf-8", errors="ignore")
        
        # Extract title
        title_match = re.search(r'#\s*ADR-\d+[:\s]+(.+)', content)
        summary = title_match.group(1).strip() if title_match else adr_file.stem
        
        decisions.append({
            "id": f"@D{num}",
            "type": "decision",
            "number": num,
            "summary": summary[:100],
            "source": f"CONTEXT/decisions/{adr_file.name}"
        })
    
    return decisions


def extract_canon_files(canon_dir: Path) -> List[Dict]:
    """Extract canon file references."""
    files = []
    if not canon_dir.exists():
        return files
    
    canon_mapping = {
        "CONTRACT.md": ("@C0", "Supreme authority - core rules"),
        "INVARIANTS.md": ("@I0", "Locked decisions index"),
        "VERSIONING.md": ("@V0", "Version control policy"),
        "GENESIS.md": ("@G0", "Bootstrap prompt"),
        "CHANGELOG.md": ("@L0", "Change log"),
        "ARBITRATION.md": ("@A0", "Conflict resolution"),
        "DEPRECATION.md": ("@P0", "Deprecation policy"),
        "MIGRATION.md": ("@M0", "Migration ceremony"),
        "CRISIS.md": ("@R0", "Crisis procedures"),
        "STEWARDSHIP.md": ("@H0", "Human escalation paths"),
    }
    
    for filename, (code, summary) in canon_mapping.items():
        filepath = canon_dir / filename
        if filepath.exists():
            files.append({
                "id": code,
                "type": "canon_file",
                "name": filename,
                "summary": summary,
                "source": f"CANON/{filename}"
            })
    
    return files


def build_codebook() -> Dict:
    """Build the complete codebook."""
    canon_dir = PROJECT_ROOT / "CANON"
    skills_dir = PROJECT_ROOT / "SKILLS"
    decisions_dir = PROJECT_ROOT / "CONTEXT" / "decisions"
    
    codebook = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "entries": {}
    }
    
    # Canon files
    for entry in extract_canon_files(canon_dir):
        codebook["entries"][entry["id"]] = entry
    
    # Contract rules
    for entry in extract_contract_rules(canon_dir / "CONTRACT.md"):
        codebook["entries"][entry["id"]] = entry
    
    # Invariants
    for entry in extract_invariants(canon_dir / "INVARIANTS.md"):
        codebook["entries"][entry["id"]] = entry
    
    # Skills
    for entry in extract_skills(skills_dir):
        codebook["entries"][entry["id"]] = entry
    
    # Decisions
    for entry in extract_decisions(decisions_dir):
        codebook["entries"][entry["id"]] = entry
    
    return codebook


def generate_markdown(codebook: Dict) -> str:
    """Generate CODEBOOK.md from codebook data."""
    lines = [
        "# Canon Codebook",
        "",
        "**Version:** {version}".format(**codebook),
        "**Generated:** {generated_at}".format(**codebook),
        "",
        "> This file is auto-generated by `TOOLS/codebook_build.py`.",
        "> Do not edit manually. Run the generator to update.",
        "",
        "---",
        "",
        "## Purpose",
        "",
        "The codebook provides stable, short IDs for referencing AGS entities.",
        "Use these IDs in prompts, documentation, and agent instructions to",
        "save tokens while maintaining precision.",
        "",
        "**Example:** Instead of \"read CANON/CONTRACT.md Rule 3\", write \"load @C0, follow @C3\".",
        "",
        "---",
        "",
    ]
    
    # Group by type
    by_type = {}
    for entry_id, entry in codebook["entries"].items():
        entry_type = entry["type"]
        if entry_type not in by_type:
            by_type[entry_type] = []
        by_type[entry_type].append(entry)
    
    # Type headers
    type_headers = {
        "canon_file": "## Canon Files (@X0)",
        "contract_rule": "## Contract Rules (@C)",
        "invariant": "## Invariants (@I)",
        "skill": "## Skills (@S)",
        "decision": "## Decisions (@D)"
    }
    
    for entry_type, header in type_headers.items():
        if entry_type not in by_type:
            continue
        
        lines.append(header)
        lines.append("")
        lines.append("| ID | Summary | Source |")
        lines.append("|----|---------|--------|")
        
        entries = sorted(by_type[entry_type], key=lambda e: e["id"])
        for entry in entries:
            entry_id = entry["id"]
            summary = entry.get("summary", entry.get("name", ""))
            source = entry.get("source", "")
            lines.append(f"| `{entry_id}` | {summary} | `{source}` |")
        
        lines.append("")
    
    # Usage section
    lines.extend([
        "---",
        "",
        "## Usage",
        "",
        "### In Prompts",
        "```",
        "Before executing, load @C0 and verify @C3.",
        "This skill requires @I1 (authority gradient) and @I2 (contract-first boot).",
        "```",
        "",
        "### In Code",
        "```python",
        "from TOOLS.codebook_lookup import lookup",
        "rule = lookup('@C3')  # Returns full rule text",
        "```",
        "",
        "### In MCP",
        "```",
        "Use tool: codebook_lookup with {\"id\": \"@C3\"}",
        "```",
        "",
        "---",
        "",
        "## Updating",
        "",
        "Regenerate this file after adding canon rules, invariants, skills, or ADRs:",
        "```bash",
        "python TOOLS/codebook_build.py",
        "```",
        "",
        "---",
        "",
        f"*Total entries: {len(codebook['entries'])}*",
        ""
    ])
    
    return "\n".join(lines)


def compute_hash(content: str) -> str:
    """Compute hash of content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Generate AGS Codebook")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check", action="store_true", help="Check if codebook needs update")
    
    args = parser.parse_args()
    
    codebook = build_codebook()
    
    if args.json:
        print(json.dumps(codebook, indent=2))
        return 0
    
    markdown = generate_markdown(codebook)
    codebook_path = PROJECT_ROOT / "CANON" / "CODEBOOK.md"
    
    if args.check:
        if codebook_path.exists():
            existing_markdown = codebook_path.read_text(encoding="utf-8")
            generated_markdown = generate_markdown(codebook)

            # Compare markdown content, ignoring timestamp and metadata
            # Extract entries section (everything between header and "Total entries")
            def extract_entries_section(md_text):
                # Remove generated_at line which changes every run
                lines = [line for line in md_text.split('\n') if 'generated_at' not in line]
                content = '\n'.join(lines)
                # Extract from first "## " to "Total entries"
                start = content.find('## Contract Rules')
                end = content.find('*Total entries:')
                if start >= 0 and end > start:
                    return content[start:end].strip()
                return content.strip()

            existing_section = extract_entries_section(existing_markdown)
            generated_section = extract_entries_section(generated_markdown)

            if compute_hash(existing_section) == compute_hash(generated_section):
                print("[codebook_build] Codebook is up to date.")
                return 0
        print("[codebook_build] Codebook needs regeneration.")
        return 1
    
    # Write codebook with provenance header
    try:
        import sys
        # Ensure PROJECT_ROOT is in path for TOOLS.provenance
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        from TOOLS.provenance import generate_header, add_header_to_content
        header = generate_header(
            generator="TOOLS/codebook_build.py",
            inputs=[
                "CANON/CONTRACT.md",
                "CANON/INVARIANTS.md",
                "SKILLS/",
                "CONTEXT/decisions/",
            ],
            output_content=markdown,
        )
        markdown = add_header_to_content(markdown, header, file_type="md")
    except ImportError as e:
        print(f"DEBUG: Provenance import failed: {e}")
        pass  # Provenance module not available, skip header
    
    codebook_path.write_text(markdown, encoding="utf-8")
    print(f"Codebook written to {codebook_path}")
    print(f"Total entries: {len(codebook['entries'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
