#!/usr/bin/env python3
"""Update canon files to use HTML comment for hash (after frontmatter)."""

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

print("Moving canon hashes to HTML comments...\n")

for filename in canon_files:
    filepath = prompts_dir / filename
    text = filepath.read_text(encoding='utf-8')
    
    # Remove sha256 from YAML frontmatter
    text_no_hash = re.sub(r'^sha256:.*\r?\n', '', text, flags=re.MULTILINE)
    
    # Compute hash of the version without the hash line
    hash_value = hashlib.sha256(text_no_hash.encode('utf-8')).hexdigest()
    
    # Add hash as HTML comment after frontmatter
    # Find end of frontmatter (second ---)
    parts = text_no_hash.split('---', 2)
    if len(parts) >= 3:
        # Reconstruct with hash comment
        new_text = f"---{parts[1]}---\n<!-- CANON_HASH: {hash_value} -->\n{parts[2]}"
        
        filepath.write_text(new_text, encoding='utf-8')
        print(f"✓ {filename}")
        print(f"  Hash: {hash_value}")
    else:
        print(f"✗ {filename} - Could not find YAML frontmatter")

print("\n✅ Canon hashes moved to HTML comments")
