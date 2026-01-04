#!/usr/bin/env python3
"""Verify canon file hashes match their frontmatter."""

import hashlib
import re
from pathlib import Path

prompts_dir = Path("NAVIGATION/PROMPTS")

canon_files = [
    "0_ORIENTATION_CANON.md",
    "1_PROMPT_POLICY_CANON.md",
    "2_PROMPT_GENERATOR_GUIDE_FINAL.md",
    "3_MASTER_PROMPT_TEMPLATE_CANON.md",
    "4_FULL_HANDOFF_TEMPLATE_CANON.md",
    "5_MINI_HANDOFF_TEMPLATE_CANON.md",
    "6_MODEL_ROUTING_CANON.md"
]

print("Verifying canon file hashes...\n")

all_valid = True
for filename in canon_files:
    filepath = prompts_dir / filename
    
    # Compute actual hash
    content = filepath.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    
    # Extract hash from frontmatter
    text = filepath.read_text(encoding='utf-8')
    match = re.search(r'sha256:\s*([a-f0-9]{64})', text)
    
    if match:
        declared_hash = match.group(1)
        matches = declared_hash == actual_hash
        status = "✓" if matches else "✗"
        
        print(f"{status} {filename}")
        print(f"  Declared: {declared_hash}")
        print(f"  Actual:   {actual_hash}")
        
        if not matches:
            all_valid = False
            print(f"  ⚠️  MISMATCH - File has been modified!")
        print()
    else:
        print(f"✗ {filename}")
        print(f"  No sha256 found in frontmatter")
        print(f"  Actual hash: {actual_hash}")
        print()
        all_valid = False

if all_valid:
    print("✅ All canon files verified - hashes match frontmatter")
else:
    print("❌ Some canon files have mismatched hashes - files may have been modified")
