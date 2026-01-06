#!/usr/bin/env python3
"""
INBOX Report Hash Tool
Computes SHA256 content hash for INBOX markdown files and inserts/updates the hash comment.
"""
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Any

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of content, excluding any existing CONTENT_HASH comment.
    
    Args:
        content: The file content
        
    Returns:
        Hex-encoded SHA256 hash
    """
    # Remove any existing CONTENT_HASH comments for hash computation
    # Match the comment and any trailing newlines (up to 2)
    cleaned_content = re.sub(
        r'<!--\s*CONTENT_HASH:\s*[a-f0-9]+\s*-->\n*',
        '',
        content,
        flags=re.IGNORECASE
    )
    
    # Compute hash of cleaned content
    return hashlib.sha256(cleaned_content.encode('utf-8')).hexdigest()


def insert_or_update_hash(filepath: Path, writer: Any = None) -> Tuple[bool, Optional[str], str]:
    """
    Insert or update the CONTENT_HASH comment in an INBOX file.
    
    Args:
        filepath: Path to the markdown file
        
    Returns:
        Tuple of (changed, old_hash, new_hash)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read current content
    content = filepath.read_text(encoding='utf-8')
    
    # Check if hash comment already exists and extract it
    hash_pattern = r'<!--\s*CONTENT_HASH:\s*([a-f0-9]+)\s*-->\n*'
    match = re.search(hash_pattern, content, re.IGNORECASE)
    
    old_hash: Optional[str] = match.group(1) if match else None
    
    # Remove any existing hash to get clean content
    clean_content = re.sub(hash_pattern, '', content, flags=re.IGNORECASE)
    
    # Compute hash of clean content
    new_hash = compute_content_hash(clean_content)
    
    # If hash hasn't changed, no need to update
    if old_hash == new_hash:
        return False, old_hash, new_hash
    
    # Insert new hash into clean content
    # Check if file starts with frontmatter
    if clean_content.startswith('---'):
        # Find end of frontmatter
        frontmatter_end = clean_content.find('---', 3)
        if frontmatter_end != -1:
            # Position right after the closing ---
            insert_pos = frontmatter_end + 3
            # Find where actual content starts (skip all whitespace)
            content_start = insert_pos
            while content_start < len(clean_content) and clean_content[content_start] in ' \t\r\n':
                content_start += 1
            # Insert hash: ---\n<!-- CONTENT_HASH: ... -->\n\n<content>
            # We keep everything up to and including the closing ---,
            # then add exactly one newline, the hash, TWO newlines, then the content
            new_content = (
                clean_content[:insert_pos] +
                '\n<!-- CONTENT_HASH: ' + new_hash + ' -->\n\n' +
                clean_content[content_start:]
            )
        else:
            # Malformed frontmatter, insert at beginning
            new_content = f'<!-- CONTENT_HASH: {new_hash} -->\n\n{clean_content}'
    else:
        # No frontmatter, insert at beginning
        new_content = f'<!-- CONTENT_HASH: {new_hash} -->\n\n{clean_content}'
    
    # Write updated content
    # Write updated content
    if not writer:
        raise RuntimeError("GuardedWriter required")
    writer.write_durable(str(filepath), new_content)
    
    return True, old_hash, new_hash


def verify_hash(filepath: Path) -> Tuple[bool, Optional[str], str]:
    """
    Verify that a file has a valid CONTENT_HASH comment.
    
    Args:
        filepath: Path to the markdown file
        
    Returns:
        Tuple of (valid, stored_hash, computed_hash)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    content = filepath.read_text(encoding='utf-8')
    
    # Extract stored hash
    hash_pattern = r'<!--\s*CONTENT_HASH:\s*([a-f0-9]+)\s*-->'
    match = re.search(hash_pattern, content, re.IGNORECASE)
    
    if not match:
        computed_hash = compute_content_hash(content)
        return False, None, computed_hash
    
    stored_hash = match.group(1)
    computed_hash = compute_content_hash(content)
    
    return stored_hash == computed_hash, stored_hash, computed_hash


def main():
    """CLI entry point."""
    # Instantiate GuardedWriter for CLI usage
    writer = None
    if GuardedWriter:
        # Assume PROJECT_ROOT is resolvable or passed?
        # This script might be run standalone.
        # Let's try to resolve project root.
        repo_root = Path(__file__).resolve().parents[4]
        writer = GuardedWriter(repo_root, durable_roots=["INBOX", "LAW/CONTRACTS/_runs"])
        writer.open_commit_gate()

    if len(sys.argv) < 2:
        print("Usage: python hash_inbox_file.py <command> <file>")
        print("Commands:")
        print("  update <file>  - Insert or update hash in file")
        print("  verify <file>  - Verify hash in file")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "update":
        if len(sys.argv) < 3:
            print("Error: update command requires a file path")
            sys.exit(1)
        
        filepath = Path(sys.argv[2])
        changed, old_hash, new_hash = insert_or_update_hash(filepath, writer)
        
        if changed:
            if old_hash:
                print(f"✅ Updated hash in {filepath}")
                print(f"   Old: {old_hash}")
                print(f"   New: {new_hash}")
            else:
                print(f"✅ Inserted hash in {filepath}")
                print(f"   Hash: {new_hash}")
        else:
            print(f"✅ Hash already up-to-date in {filepath}")
            print(f"   Hash: {new_hash}")
    
    elif command == "verify":
        if len(sys.argv) < 3:
            print("Error: verify command requires a file path")
            sys.exit(1)
        
        filepath = Path(sys.argv[2])
        valid, stored_hash, computed_hash = verify_hash(filepath)
        
        if valid:
            print(f"✅ Hash valid in {filepath}")
            print(f"   Hash: {stored_hash}")
            sys.exit(0)
        else:
            if stored_hash:
                print(f"❌ Hash mismatch in {filepath}")
                print(f"   Stored:   {stored_hash}")
                print(f"   Computed: {computed_hash}")
            else:
                print(f"❌ No hash found in {filepath}")
                print(f"   Computed: {computed_hash}")
            sys.exit(1)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Valid commands: update, verify")
        sys.exit(1)


if __name__ == "__main__":
    main()
