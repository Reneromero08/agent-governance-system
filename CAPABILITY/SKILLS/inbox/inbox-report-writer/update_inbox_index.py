#!/usr/bin/env python3
"""
INBOX Index Updater
Automatically updates INBOX/INBOX.md with current file listings and metadata.
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from hash_inbox_file import verify_hash
from generate_inbox_ledger import extract_frontmatter, list_inbox_markdown_files

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def get_file_entry(filepath: Path, inbox_root: Path) -> dict:
    """Get metadata for a single file to include in INBOX.md."""
    try:
        content = filepath.read_text(encoding='utf-8')
        frontmatter = extract_frontmatter(content)
        valid, stored_hash, computed_hash = verify_hash(filepath)
        
        stat = filepath.stat()
        rel_path = filepath.relative_to(inbox_root)
        
        return {
            'path': str(rel_path).replace('\\', '/'),
            'filename': filepath.name,
            'section': frontmatter.get('section', 'unknown'),
            'title': frontmatter.get('title', filepath.stem),
            'author': frontmatter.get('author', 'Unknown'),
            'priority': frontmatter.get('priority', 'Medium'),
            'created': frontmatter.get('created', 'UNKNOWN'),
            'modified': frontmatter.get('modified', datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')),
            'summary': frontmatter.get('summary', ''),
            'hash_valid': valid,
            'hash': {
                'stored': stored_hash,
                'computed': computed_hash
            }
        }
    except Exception as e:
        return None


def generate_inbox_index(inbox_path: Path = None) -> str:
    """Generate the INBOX.md index content."""
    if inbox_path is None:
        inbox_path = Path.cwd() / "INBOX"
    
    # Find all markdown files except INBOX.md and LEDGER.yaml
    md_files = [
        f for f in list_inbox_markdown_files(inbox_path)
        if f.name not in ['INBOX.md', 'LEDGER.yaml'] and not f.name.startswith('.')
    ]
    
    # Organize by category
    categories = {}
    for md_file in md_files:
        entry = get_file_entry(md_file, inbox_path)
        if entry:
            section = entry['section']
            if section not in categories:
                categories[section] = []
            categories[section].append(entry)
    
    # Sort within categories by created date (newest first)
    for section in categories:
        categories[section].sort(key=lambda x: x.get('created', ''), reverse=True)
    
    # Generate markdown content
    lines = []
    lines.append("<!-- CONTENT_HASH: PLACEHOLDER -->")
    lines.append("")
    lines.append("# INBOX")
    lines.append("")
    lines.append("This directory stores new research, reports, and roadmaps for user approval.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Add entries by section
    section_order = ['report', 'guide', 'roadmap', 'research', 'agents', 'unknown']
    
    for section in section_order:
        if section not in categories:
            continue
        
        entries = categories[section]
        if not entries:
            continue
        
        lines.append("")
        
        for entry in entries:
            # Format: - [ ] **[title](path)**
            hash_indicator = "✅" if entry['hash_valid'] else "⚠️"
            hash_short = entry.get('hash', {}).get('stored', 'none')[:8] if entry.get('hash', {}).get('stored') else 'missing'
            lines.append(f" - [ ] **[{entry['filename']}]({entry['path']})** {hash_indicator} `{hash_short}...`")
            lines.append(f"   - **Section:** #{entry['section']}")
            lines.append(f"   - **Author:** {entry['author']}")
            lines.append(f"   - **Priority:** {entry['priority']}")
            lines.append(f"   - **Created:** {entry['created']}")
            lines.append(f"   - **Modified:** {entry['modified']}")
            if entry['summary']:
                lines.append(f"   - **Summary:** {entry['summary']}")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("# Process")
    lines.append("")
    lines.append("1. **Inbox Entry:** All incoming docs start here.")
    lines.append("2. **Metadata:** Every entry MUST have a list item in the sections above.")
    lines.append("3. **Hash:** Every document MUST start with `<!-- CONTENT_HASH: ... -->` (Line 1).")
    lines.append("4. **Approval:** Governance checks hash, User approves content.")
    lines.append("5. **Promotion:** Move to permanent location (CANON, CONTEXT, etc).")
    lines.append("")
    lines.append("## Authority Rule")
    lines.append("")
    lines.append("No document bypasses INBOX. Hash integrity is enforced by pre-commit hooks.")
    lines.append("")
    
    return '\n'.join(lines)


def update_inbox_index(inbox_path: Path = None, quiet: bool = False, writer: Any = None) -> bool:
    """
    Update INBOX/INBOX.md with current file listings.
    
    Returns:
        True if updated, False if no changes needed
    """
    if inbox_path is None:
        inbox_path = Path.cwd() / "INBOX"
    
    inbox_md = inbox_path / "INBOX.md"
    
    # Generate new content
    new_content = generate_inbox_index(inbox_path)
    
    # Write to file
    # Write to file
    if not writer:
         raise RuntimeError("GuardedWriter required")
    
    writer.write_durable(str(inbox_md), new_content)
    
    # Now update the hash
    from hash_inbox_file import insert_or_update_hash
    changed, old_hash, new_hash = insert_or_update_hash(inbox_md, writer)
    
    if not quiet:
        if changed:
            # Count indexed files
            file_count = len([line for line in new_content.split('\n') if line.strip().startswith('- [ ]')])
            print(f"✅ Updated INBOX.md index")
            print(f"   Files indexed: {file_count}")
        else:
            print(f"✅ INBOX.md index already up-to-date")
    
    return changed


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update INBOX.md index')
    parser.add_argument('--inbox', type=Path, help='Path to INBOX directory (default: ./INBOX)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output messages')
    
    args = parser.parse_args()
    
    try:
        writer = None
        if GuardedWriter:
             repo_root = Path(__file__).resolve().parents[4]
             writer = GuardedWriter(repo_root, durable_roots=["INBOX", "LAW/CONTRACTS/_runs"])
             writer.open_commit_gate()

        update_inbox_index(args.inbox, args.quiet, writer)
        return 0
    except Exception as e:
        if not args.quiet:
            print(f"❌ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
