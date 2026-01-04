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
    text = filepath.read_text(encoding='utf-8')

    # Compute hash of content EXCLUDING the CANON_HASH line
    # Remove the entire HTML comment line containing the hash
    lines = text.split('\n')
    lines_without_hash = [line for line in lines if not re.match(r'<!-- CANON_HASH:', line)]
    content_without_hash = '\n'.join(lines_without_hash)

    # Compute hash of content without the hash line
    actual_hash = hashlib.sha256(content_without_hash.encode('utf-8')).hexdigest()

    # Update CANON_HASH in the HTML comment
    updated_text = re.sub(
        r'(<!-- CANON_HASH:\s*)[a-f0-9]{64}(\s*-->)',
        f'\\g<1>{actual_hash}\\g<2>',
        text
    )
    
    if updated_text != text:
        filepath.write_text(updated_text, encoding='utf-8')
        print(f"[OK] Updated {filename}")
        print(f"  New hash: {actual_hash}")
    else:
        print(f"[SKIP] {filename} (no change needed)")

    print("\n[OK] All canon frontmatter hashes updated")
