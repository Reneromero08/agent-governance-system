#!/usr/bin/env python3
"""Update canon hashes in all prompt files."""

import re
from pathlib import Path

# Current canon hashes (final - after moving to HTML comments)
POLICY_HASH = "6b3d1a1c6062aaa11b348a76d2ea6d39760927917b6d26a75ef4d413847ab516"
GUIDE_HASH = "cef08c3510f5420f69c228d32d01d5b8bd74d4339f7b800598cf9ae4fbdb0247"



# Find all prompt files
prompts_dir = Path("NAVIGATION/PROMPTS")
prompt_files = list(prompts_dir.glob("PHASE_*/*.md"))

print(f"Found {len(prompt_files)} prompt files to update")

updated_count = 0
for prompt_file in sorted(prompt_files):
    content = prompt_file.read_text(encoding='utf-8')
    original_content = content
    
    # Update policy_canon_sha256
    content = re.sub(
        r'policy_canon_sha256:\s*"[^"]*"',
        f'policy_canon_sha256: "{POLICY_HASH}"',
        content
    )
    
    # Update guide_canon_sha256
    content = re.sub(
        r'guide_canon_sha256:\s*"[^"]*"',
        f'guide_canon_sha256: "{GUIDE_HASH}"',
        content
    )
    if content != original_content:
        prompt_file.write_text(content, encoding='utf-8')
        updated_count += 1
        print(f"✓ Updated: {prompt_file.relative_to(prompts_dir)}")

print(f"\n✅ Updated {updated_count} prompt files")
