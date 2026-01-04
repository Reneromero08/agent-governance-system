#!/usr/bin/env python3
"""Update canon hashes in all prompt files."""

import re
from pathlib import Path

# Current canon hashes (final - after moving to HTML comments)
POLICY_HASH = "29ed1cec0104314dea9bb5844e9fd7c15a162313ef7cc3a19e8b898d9cea2624"
GUIDE_HASH = "9acc1b9772579720e3c8bc19a80a9a3908323b411c6d06d04952323668f4efe4"



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
