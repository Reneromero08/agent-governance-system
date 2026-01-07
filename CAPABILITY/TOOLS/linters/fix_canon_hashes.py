#!/usr/bin/env python3
"""Update canon files to use HTML comment for hash with dry-run safety."""

import argparse
import hashlib
import re
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

prompts_dir = REPO_ROOT / "NAVIGATION" / "PROMPTS"

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
    parser = argparse.ArgumentParser(description="Move canon hashes to HTML comments")
    parser.add_argument("--apply", action="store_true",
                       help="Apply changes (default: dry-run only)")
    args = parser.parse_args()

    # Initialize GuardedWriter with CANON exemption for linters
    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["NAVIGATION/PROMPTS", "LAW/CANON"]  # EXEMPTION: Linters only
    )

    print("Moving canon hashes to HTML comments...\n")

    if not args.apply:
        print("[DRY-RUN] Showing changes that would be applied\n")

    changes_detected = False
    for filename in canon_files:
        filepath = prompts_dir / filename
        
        # Skip if file doesn't exist
        if not filepath.exists():
            print(f"[SKIP] {filename} (file not found)")
            continue
        
        text = filepath.read_text(encoding='utf-8')
        
        # Check if already converted
        if '<!-- CANON_HASH:' in text:
            print(f"[SKIP] {filename} (already converted)")
            continue
        
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
            
            changes_detected = True
            if args.apply:
                # Open commit gate and write with audit trail
                writer.open_commit_gate()
                rel_path = filepath.relative_to(REPO_ROOT)
                writer.write_durable(str(rel_path), new_text)
                print(f"[APPLIED] {filename}")
                print(f"  Hash: {hash_value}")
            else:
                print(f"[DRY-RUN] Would update: {filename}")
                print(f"  Hash: {hash_value}")
        else:
            print(f"[ERROR] {filename} - Could not find YAML frontmatter")

    if not args.apply:
        if changes_detected:
            print("\nTo apply these changes, run with --apply flag")
            return 1
        else:
            print("\n[OK] All canon files are already converted")
            return 0

    print("\n✅ Canon hashes moved to HTML comments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
