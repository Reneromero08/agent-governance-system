#!/usr/bin/env python3
"""
Pre-commit check for INBOX file hashes.
Validates that all staged INBOX/*.md files have valid content hashes.
"""
import subprocess
import sys
from pathlib import Path

# Import the hash verification function
sys.path.insert(0, str(Path(__file__).parent.parent))
from hash_inbox_file import verify_hash


def get_staged_inbox_files():
    """Get list of staged INBOX markdown files."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
            capture_output=True,
            text=True,
            check=True
        )
        
        files = result.stdout.strip().split('\n')
        inbox_files = [
            f for f in files 
            if f.startswith('INBOX/') and f.endswith('.md')
        ]
        
        return inbox_files
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting staged files: {e}")
        return []


def main():
    """Check all staged INBOX files for valid hashes."""
    staged_files = get_staged_inbox_files()
    
    if not staged_files:
        # No INBOX files staged, pass
        sys.exit(0)
    
    print(f"🔍 Checking {len(staged_files)} staged INBOX file(s)...")
    
    errors = []
    
    for filepath_str in staged_files:
        filepath = Path(filepath_str)
        
        if not filepath.exists():
            # File was deleted, skip
            continue
        
        try:
            valid, stored_hash, computed_hash = verify_hash(filepath)
            
            if not valid:
                if stored_hash:
                    errors.append(
                        f"  ❌ {filepath}: Hash mismatch\n"
                        f"     Stored:   {stored_hash}\n"
                        f"     Computed: {computed_hash}"
                    )
                else:
                    errors.append(
                        f"  ❌ {filepath}: Missing CONTENT_HASH comment\n"
                        f"     Computed: {computed_hash}"
                    )
            else:
                print(f"  ✅ {filepath}: Hash valid")
        
        except Exception as e:
            errors.append(f"  ❌ {filepath}: Error - {e}")
    
    if errors:
        print("\n❌ INBOX hash validation failed!")
        print("\nErrors:")
        for error in errors:
            print(error)
        print("\nTo fix:")
        print("  python CAPABILITY/SKILLS/inbox/inbox-report-writer/hash_inbox_file.py update <file>")
        print("\nThen stage the updated file and commit again.")
        sys.exit(1)
    
    print("✅ All INBOX files have valid hashes")
    sys.exit(0)


if __name__ == "__main__":
    main()
