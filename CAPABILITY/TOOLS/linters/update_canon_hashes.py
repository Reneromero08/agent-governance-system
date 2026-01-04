#!/usr/bin/env python3
"""Update canon file frontmatter hashes to match actual content."""

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

print("Updating canon file frontmatter hashes...\n")

for filename in canon_files:
    filepath = prompts_dir / filename
    
    # Read content
    content = filepath.read_bytes()
    text = filepath.read_text(encoding='utf-8')
    
    # Compute hash (excluding the hash line itself for consistency)
    # We'll compute hash, update the file, then recompute
    actual_hash = hashlib.sha256(content).hexdigest()
    
    # Update sha256 in frontmatter
    updated_text = re.sub(
        r'(sha256:\s*)[a-f0-9]{64}',
        f'\\g<1>{actual_hash}',
        text
    )
    
    if updated_text != text:
        filepath.write_text(updated_text, encoding='utf-8')
        print(f"✓ Updated {filename}")
        print(f"  New hash: {actual_hash}")
    else:
        print(f"- {filename} (no change needed)")
    
print("\n✅ All canon frontmatter hashes updated")
