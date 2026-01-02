
import os
import re
from pathlib import Path

# Mapping of old top-level modules to new buckets
# "OLD_MODULE": "NEW_BUCKET.OLD_MODULE"
MOVES = {
    "CANON": "LAW.CANON",
    "CONTRACTS": "LAW.CONTRACTS",
    "SKILLS": "CAPABILITY.SKILLS",
    "TOOLS": "CAPABILITY.TOOLS",
    "MCP": "CAPABILITY.MCP",
    "PRIMITIVES": "CAPABILITY.PRIMITIVES",
    "PIPELINES": "CAPABILITY.PIPELINES",
    "CORTEX": "NAVIGATION.CORTEX",
    "CONTEXT.maps": "NAVIGATION.maps",
    # THOUGHT/research imports? usually strictly documents.
}

REPO_ROOT = Path(".").resolve()
DRY_RUN = False

def refactor_imports():
    print(f"Starting Import Refactor in {REPO_ROOT}")
    
    count = 0
    files_changed = 0
    
    for python_file in REPO_ROOT.rglob("*.py"):
        # Skip venv or .git if any
        if ".venv" in str(python_file) or ".git" in str(python_file):
            continue
            
        original_content = python_file.read_text(encoding="utf-8")
        new_content = original_content
        
        # Regex replacement
        # We need to be careful. 
        # "from NAVIGATION.CORTEX.db" -> "from NAVIGATION.CORTEX.db"
        # "import NAVIGATION.CORTEX" -> "import NAVIGATION.CORTEX" (Unlikely used, usually from x import y)
        
        for old, new in MOVES.items():
            # Pattern 1: from OLD import ...
            # strict word boundary on left, dot or space on right
            pattern_from = fr'from {old}\b'
            replacement_from = f'from {new}'
            
            # Pattern 2: import OLD
            pattern_import = fr'import {old}\b'
            replacement_import = f'import {new}'
            
            new_content = re.sub(pattern_from, replacement_from, new_content)
            new_content = re.sub(pattern_import, replacement_import, new_content)
            
        if new_content != original_content:
            print(f"Patching {python_file.relative_to(REPO_ROOT)}")
            if not DRY_RUN:
                python_file.write_text(new_content, encoding="utf-8")
            files_changed += 1
            
    print(f"Refactor Complete. {files_changed} files updated.")

if __name__ == "__main__":
    refactor_imports()
