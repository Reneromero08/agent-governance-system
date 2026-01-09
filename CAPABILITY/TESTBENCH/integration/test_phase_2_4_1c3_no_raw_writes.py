#!/usr/bin/env python3
"""
Phase 2.4.1C.3 No Raw Writes Test

Mechanically scans CORTEX/** and CAPABILITY/SKILLS/** for raw write operations.
Fails if any direct filesystem operations are found outside adapter files.
"""

import os
import re
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


# Patterns that indicate raw write operations
RAW_WRITE_PATTERNS = [
    r'\.write_text\s*\(',
    r'\.write_bytes\s*\(',
    r'\.open\s*\(',
    r'\.mkdir\s*\(',
    r'\.rename\s*\(',
    r'\.replace\s*\(',
    r'\.unlink\s*\(',
    r'shutil\.(copy|copy2|move|rmtree)',
    r'os\.(remove|rename|replace)'
]

# Files that are allowed to contain raw writes (adapters, utilities)
ALLOWED_FILES = {
    'guarded_writer.py',
    'write_firewall.py',
    'test_ant_worker.py',
    'test_inbox_hash.py',
    'test_doc_merge_batch.py',
    'init_skill.py',  # Skill creation utility - needs raw writes to bootstrap new skills
    'package_skill.py',  # Skill packaging utility - needs raw writes to create .skill files
    'workspace_isolation.py',  # Worktree management utility - string operations flagged as false positives
    'inbox_normalize.py',  # Report normalization - needs legitimate file operations
    'weekly_normalize.py',  # Report normalization - needs legitimate file operations
    'cleanup_report_formatting.py'  # Report cleanup - needs legitimate file operations
}

# Lines that should be ignored (comments, imports, defensive code)
IGNORE_PATTERNS = [
    r'#.*',
    r'""".*"""',
    r"'''.*'''",
    r'except.*:',
    r'raise.*:',
    r'import.*',
    r'from.*import.*'
]


def is_safe_line(line: str, filepath: str) -> bool:
    """Check if a line should be ignored (safe from raw write violations)."""
    line_stripped = line.strip()

    # Skip if file is allowed
    filename = Path(filepath).name
    if filename in ALLOWED_FILES:
        return True

    # Skip if line matches ignore patterns
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    # Skip if it's just a string replace operation (not filesystem replace)
    if '.replace(' in line and not any(x in line for x in ['Path', 'path.', 'os.', 'shutil.']):
        return True

    # Skip if .open() is in read mode (contains "r", "rb", or read-related flags)
    if '.open(' in line:
        # Check for read mode indicators: "r", "rb", "r+", etc.
        if re.search(r'''\.open\s*\(\s*["']r[bt]?["']''', line):
            return True

    return False


def scan_for_raw_writes(directory: Path) -> list:
    """Scan directory for raw write operations."""
    violations = []
    
    if not directory.exists():
        return violations
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            filename = Path(filepath).name
            
            # Skip allowed files
            if filename in ALLOWED_FILES:
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    # Check each raw write pattern
                    for pattern in RAW_WRITE_PATTERNS:
                        if re.search(pattern, line):
                            # Skip if it's a safe line
                            if is_safe_line(line, filepath):
                                continue
                                
                            violations.append({
                                'file': filepath,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
                            break  # Only report once per line
                            
            except Exception as e:
                violations.append({
                    'file': filepath,
                    'line': 0,
                    'content': f'Error reading file: {e}',
                    'pattern': 'READ_ERROR'
                })
    
    return violations


def test_no_raw_writes_in_cortex_and_skills():
    """Test that CORTEX/** and CAPABILITY/SKILLS/** contain no raw write operations."""
    cortex_dir = REPO_ROOT / "NAVIGATION" / "CORTEX"
    skills_dir = REPO_ROOT / "CAPABILITY" / "SKILLS"
    
    cortex_violations = scan_for_raw_writes(cortex_dir)
    skills_violations = scan_for_raw_writes(skills_dir)
    
    all_violations = cortex_violations + skills_violations
    
    # Create compact list for assertion message
    violation_list = []
    for v in all_violations:
        rel_path = str(Path(v['file']).relative_to(REPO_ROOT))
        violation_list.append(f"{rel_path}:{v['line']}")
    
    # Fail if any violations exist
    assert not all_violations, f"Raw write violations found ({len(all_violations)}): {', '.join(violation_list[:20])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
