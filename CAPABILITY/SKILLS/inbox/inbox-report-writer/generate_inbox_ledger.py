#!/usr/bin/env python3
"""
INBOX Ledger Generator
Scans INBOX directory and generates a YAML ledger with metadata for all files.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Any
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from hash_inbox_file import verify_hash

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SECTION_INDEX_PATHS = [
    PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated" / "SECTION_INDEX.json",
    PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "meta" / "SECTION_INDEX.json",
]


def extract_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith('---'):
        return {}
    
    end = content.find('---', 3)
    if end == -1:
        return {}
    
    frontmatter_text = content[3:end].strip()
    try:
        return yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return {}


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _load_section_index() -> List[dict]:
    for index_path in SECTION_INDEX_PATHS:
        if index_path.exists():
            return json.loads(index_path.read_text(encoding="utf-8"))
    raise FileNotFoundError("SECTION_INDEX.json not found in cortex outputs")


def list_inbox_markdown_files(inbox_root: Path) -> List[Path]:
    """List INBOX markdown files using the cortex index."""
    records = _load_section_index()
    paths = set()
    for record in records:
        record_path = str(record.get("path", "")) if "path" in record else ""
        anchor = str(record.get("anchor", "")) if "anchor" in record else ""
        anchor_path = anchor.split("#", 1)[0] if anchor else ""
        candidate = record_path or anchor_path
        normalized = _normalize_path(candidate)
        if not normalized.startswith("INBOX/") or not normalized.endswith(".md"):
            continue
        paths.add(normalized)

    results: List[Path] = []
    for rel_path in sorted(paths):
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists() and full_path.is_file():
            results.append(full_path)
    return results


def get_file_metadata(filepath: Path, inbox_root: Path) -> dict:
    """Get metadata for a single INBOX file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        
        # Extract frontmatter
        frontmatter = extract_frontmatter(content)
        
        # Verify hash
        valid, stored_hash, computed_hash = verify_hash(filepath)
        
        # Get file stats
        stat = filepath.stat()
        
        # Determine relative path from INBOX
        try:
            rel_path = filepath.relative_to(inbox_root)
        except ValueError:
            rel_path = filepath
        
        metadata = {
            'path': str(rel_path).replace('\\', '/'),
            'filename': filepath.name,
            'size_bytes': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'hash': {
                'valid': valid,
                'stored': stored_hash,
                'computed': computed_hash,
                'match': stored_hash == computed_hash if stored_hash else False
            }
        }
        
        # Add frontmatter fields if present
        if frontmatter:
            metadata['frontmatter'] = {
                'title': frontmatter.get('title'),
                'section': frontmatter.get('section'),
                'author': frontmatter.get('author'),
                'priority': frontmatter.get('priority'),
                'status': frontmatter.get('status'),
                'created': frontmatter.get('created'),
                'tags': frontmatter.get('tags', [])
            }
            # Remove None values
            metadata['frontmatter'] = {k: v for k, v in metadata['frontmatter'].items() if v is not None}
        
        return metadata
    
    except Exception as e:
        return {
            'path': str(filepath),
            'filename': filepath.name,
            'error': str(e)
        }


def scan_inbox_directory(inbox_path: Path) -> dict:
    """Scan INBOX directory and collect metadata for all markdown files."""
    if not inbox_path.exists():
        raise FileNotFoundError(f"INBOX directory not found: {inbox_path}")
    
    # Find all markdown files
    md_files = list_inbox_markdown_files(inbox_path)
    
    # Organize by subdirectory
    ledger = {
        'generated': datetime.now().isoformat(),
        'inbox_path': str(inbox_path),
        'total_files': len(md_files),
        'summary': {
            'valid_hashes': 0,
            'invalid_hashes': 0,
            'missing_hashes': 0,
            'errors': 0
        },
        'files_by_category': {}
    }
    
    for md_file in md_files:
        metadata = get_file_metadata(md_file, inbox_path)
        
        # Update summary
        if 'error' in metadata:
            ledger['summary']['errors'] += 1
        elif metadata['hash']['valid']:
            ledger['summary']['valid_hashes'] += 1
        elif metadata['hash']['stored']:
            ledger['summary']['invalid_hashes'] += 1
        else:
            ledger['summary']['missing_hashes'] += 1
        
        # Categorize by subdirectory
        try:
            rel_path = md_file.relative_to(inbox_path)
            if len(rel_path.parts) > 1:
                category = rel_path.parts[0]
            else:
                category = 'root'
        except ValueError:
            category = 'unknown'
        
        if category not in ledger['files_by_category']:
            ledger['files_by_category'][category] = []
        
        ledger['files_by_category'][category].append(metadata)
    
    # Sort files within each category by path
    for category in ledger['files_by_category']:
        ledger['files_by_category'][category].sort(key=lambda x: x.get('path', ''))
    
    return ledger


def generate_ledger(inbox_path: Path = None, output_path: Path = None, quiet: bool = False, writer: Any = None) -> str:
    """
    Generate INBOX ledger and write to file.
    
    Args:
        inbox_path: Path to INBOX directory (default: ./INBOX)
        output_path: Path to output YAML file (default: INBOX/LEDGER.yaml)
        quiet: Suppress output messages
        writer: Optional GuardedWriter instance
    
    Returns:
        Path to generated ledger file
    """
    if inbox_path is None:
        inbox_path = Path.cwd() / "INBOX"
    
    if output_path is None:
        output_path = inbox_path / "LEDGER.yaml"
    
    if not quiet:
        print(f"📊 Scanning INBOX directory: {inbox_path}")
    ledger = scan_inbox_directory(inbox_path)
    
    if not quiet:
        print(f"📝 Found {ledger['total_files']} markdown files")
        print(f"   ✅ Valid hashes: {ledger['summary']['valid_hashes']}")
        print(f"   ❌ Invalid hashes: {ledger['summary']['invalid_hashes']}")
        print(f"   ⚠️  Missing hashes: {ledger['summary']['missing_hashes']}")
        if ledger['summary']['errors'] > 0:
            print(f"   🔥 Errors: {ledger['summary']['errors']}")
    
    # Write YAML
    if not writer:
        raise RuntimeError("GuardedWriter required")
    import io
    stream = io.StringIO()
    yaml.dump(ledger, stream, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Use write_tmp for paths in _tmp directories, write_durable for others
    output_str = str(output_path)
    if "_tmp" in output_str or "\\tmp\\" in output_str or "/tmp/" in output_str:
        writer.write_tmp(output_str, stream.getvalue())
    else:
        writer.write_durable(output_str, stream.getvalue())
    
    if not quiet:
        print(f"✅ Ledger written to: {output_path}")
    
    return str(output_path)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate INBOX ledger')
    parser.add_argument('--inbox', type=Path, help='Path to INBOX directory (default: ./INBOX)')
    parser.add_argument('--output', type=Path, help='Output YAML file (default: INBOX/LEDGER.yaml)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output messages')
    parser.add_argument('--watch', action='store_true', help='Watch for changes and regenerate (not implemented yet)')
    
    args = parser.parse_args()
    
    try:
        # Instantiate GuardedWriter for CLI usage
        writer = None
        if GuardedWriter:
             repo_root = Path(__file__).resolve().parents[4]
             writer = GuardedWriter(repo_root, durable_roots=["INBOX", "LAW/CONTRACTS/_runs"])
             writer.open_commit_gate()

        generate_ledger(args.inbox, args.output, args.quiet, writer)
        return 0
    except Exception as e:
        if not args.quiet:
            print(f"❌ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
