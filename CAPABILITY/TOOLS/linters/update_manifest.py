#!/usr/bin/env python3
"""Compute and update all canon hashes in manifest with dry-run safety."""

import argparse
import json
import hashlib
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

prompts_dir = REPO_ROOT / "NAVIGATION" / "PROMPTS"
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


def main():
    parser = argparse.ArgumentParser(description="Update manifest with canon hashes")
    parser.add_argument("--apply", action="store_true",
                       help="Apply changes (default: dry-run only)")
    args = parser.parse_args()

    # Initialize GuardedWriter with NAVIGATION/PROMPTS for manifest updates
    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["NAVIGATION/PROMPTS", "LAW/CANON"]  # EXEMPTION: Linters only
    )

    print("Computing canon hashes...")
    hashes = {}
    for filename in canon_files:
        filepath = prompts_dir / filename
        if filepath.exists():
            content = filepath.read_bytes()
            hash_value = hashlib.sha256(content).hexdigest()
            hashes[filename] = hash_value
            print(f"  {filename}: {hash_value}")

    if not args.apply:
        print("\n[DRY-RUN] Manifest changes that would be applied:")
        if manifest_path.exists():
            print(f"  Updated linter path to: CAPABILITY/TOOLS/linters/lint_prompt_pack.sh")
            print(f"  Updated all canon hashes in manifest")
            print(f"\nTo apply these changes, run with --apply flag")
            return 0
        else:
            print("  Manifest file not found")
            print(f"\nCannot apply changes - manifest file does not exist")
            return 1

    # Write manifest with audit trail only if manifest exists
    if not manifest_path.exists():
        print(f"\n[ERROR] Manifest file not found: {manifest_path}")
        return 1

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

    # Write manifest with audit trail
    writer.open_commit_gate()
    rel_path = manifest_path.relative_to(REPO_ROOT)
    writer.write_durable(str(rel_path), json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\n✅ Updated manifest: {manifest_path}")
    print(f"✅ Updated linter path to: CAPABILITY/TOOLS/linters/lint_prompt_pack.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
