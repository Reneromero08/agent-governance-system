#!/usr/bin/env python3
"""Update canon file frontmatter hashes to match actual content with dry-run safety."""

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
    parser = argparse.ArgumentParser(description="Update canon file frontmatter hashes")
    parser.add_argument("--apply", action="store_true",
                       help="Apply changes (default: dry-run only)")
    args = parser.parse_args()

    # Initialize GuardedWriter with CANON exemption for linters
    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["NAVIGATION/PROMPTS", "LAW/CANON"]  # EXEMPTION: Linters only
    )

    print("Updating canon file frontmatter hashes...\n")

    if not args.apply:
        print("[DRY-RUN] Showing changes that would be applied\n")

    changes_detected = False
    for filename in canon_files:
        filepath = prompts_dir / filename
        
        # Skip if file doesn't exist
        if not filepath.exists():
            print(f"[SKIP] {filename} (file not found)")
            continue

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
            changes_detected = True
            if args.apply:
                # Open commit gate and write with audit trail
                writer.open_commit_gate()
                rel_path = filepath.relative_to(REPO_ROOT)
                writer.write_durable(str(rel_path), updated_text)
                print(f"[APPLIED] {filename}")
                print(f"  New hash: {actual_hash}")
            else:
                print(f"[DRY-RUN] Would update: {filename}")
                print(f"  New hash: {actual_hash}")
        else:
            print(f"[SKIP] {filename} (no change needed)")

    if not args.apply:
        if changes_detected:
            print("\nTo apply these changes, run with --apply flag")
            return 1
        else:
            print("\n[OK] All canon frontmatter hashes are current")
            return 0

    print("\n[OK] All canon frontmatter hashes updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
