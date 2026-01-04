#!/usr/bin/env python3
"""Compute and update all canon hashes in manifest."""

import json
import hashlib
from pathlib import Path

prompts_dir = Path("NAVIGATION/PROMPTS")
manifest_path = prompts_dir / "PROMPT_PACK_MANIFEST.json"

# Compute hashes for all canon files
canon_files = [
    "0_ORIENTATION_CANON.md",
    "1_PROMPT_POLICY_CANON.md",
    "2_PROMPT_GENERATOR_GUIDE_FINAL.md",
    "3_MASTER_PROMPT_TEMPLATE_CANON.md",
    "4_FULL_HANDOFF_TEMPLATE_CANON.md",
    "5_MINI_HANDOFF_TEMPLATE_CANON.md",
    "6_MODEL_ROUTING_CANON.md"
]

print("Computing canon hashes...")
hashes = {}
for filename in canon_files:
    filepath = prompts_dir / filename
    if filepath.exists():
        content = filepath.read_bytes()
        hash_value = hashlib.sha256(content).hexdigest()
        hashes[filename] = hash_value
        print(f"  {filename}: {hash_value}")

# Load manifest
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

# Update hashes
manifest["canon"]["orientation"]["sha256"] = hashes["0_ORIENTATION_CANON.md"]
manifest["canon"]["policy"]["sha256"] = hashes["1_PROMPT_POLICY_CANON.md"]
manifest["canon"]["guide"]["sha256"] = hashes["2_PROMPT_GENERATOR_GUIDE_FINAL.md"]
manifest["canon"]["template"]["sha256"] = hashes["3_MASTER_PROMPT_TEMPLATE_CANON.md"]
manifest["canon"]["handoff_full"]["sha256"] = hashes["4_FULL_HANDOFF_TEMPLATE_CANON.md"]
manifest["canon"]["handoff_mini"]["sha256"] = hashes["5_MINI_HANDOFF_TEMPLATE_CANON.md"]
manifest["canon"]["routing"]["sha256"] = hashes["6_MODEL_ROUTING_CANON.md"]

# Update linter path
manifest["lint_script"]["path"] = "CAPABILITY/TOOLS/linters/lint_prompt_pack.sh"
manifest["lint_script"]["status"] = "operational"

# Write back
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

print(f"\n✅ Updated manifest: {manifest_path}")
print(f"✅ Updated linter path to: CAPABILITY/TOOLS/linters/lint_prompt_pack.sh")
