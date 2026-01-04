#!/usr/bin/env python3
"""
INBOX Ledger Generator
Scans INBOX directory and generates a YAML ledger with metadata for all files.
"""
import sys
from pathlib import Path
from datetime import datetime
import yaml
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from hash_inbox_file import verify_hash, compute_content_hash


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


def get_file_metadata(filepath: Path) -> dict:
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
            rel_path = filepath.relative_to(Path.cwd() / "INBOX")
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
    md_files = list(inbox_path.rglob("*.md"))
    
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
        metadata = get_file_metadata(md_file)
        
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


def generate_ledger(inbox_path: Path = None, output_path: Path = None, quiet: bool = False) -> str:
    """
    Generate INBOX ledger and write to file.
    
    Args:
        inbox_path: Path to INBOX directory (default: ./INBOX)
        output_path: Path to output YAML file (default: INBOX/LEDGER.yaml)
        quiet: Suppress output messages
    
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
    with output_path.open('w', encoding='utf-8') as f:
        yaml.dump(ledger, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
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
        generate_ledger(args.inbox, args.output, args.quiet)
        return 0
    except Exception as e:
        if not args.quiet:
            print(f"❌ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
