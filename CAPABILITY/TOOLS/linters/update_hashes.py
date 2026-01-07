#!/usr/bin/env python3
"""Update canon hashes in all prompt files with dry-run safety."""

import argparse
import re
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

# Current canon hashes (final - after moving to HTML comments)
POLICY_HASH = "29ed1cec0104314dea9bb5844e9fd7c15a162313ef7cc3a19e8b898d9cea2624"
GUIDE_HASH = "9acc1b9772579720e3c8bc19a80a9a3908323b411c6d06d04952323668f4efe4"


def main():
    parser = argparse.ArgumentParser(description="Update canon hashes in prompt files")
    parser.add_argument("--apply", action="store_true",
                       help="Apply changes (default: dry-run only)")
    args = parser.parse_args()

    # Initialize GuardedWriter with CANON exemption for linters
    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["NAVIGATION/PROMPTS", "LAW/CANON"]  # EXEMPTION: Linters only
    )

    # Find all prompt files
    prompts_dir = REPO_ROOT / "NAVIGATION" / "PROMPTS"
    prompt_files = list(prompts_dir.glob("PHASE_*/*.md"))

    print(f"Found {len(prompt_files)} prompt files to update")

    if not args.apply:
        print("[DRY-RUN] Showing changes that would be applied\n")

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
            if args.apply:
                # Open commit gate and write with audit trail
                writer.open_commit_gate()
                rel_path = prompt_file.relative_to(REPO_ROOT)
                writer.write_durable(str(rel_path), content)
                print(f"[APPLIED] {prompt_file.relative_to(prompts_dir)}")
            else:
                print(f"[DRY-RUN] Would update: {prompt_file.relative_to(prompts_dir)}")
            updated_count += 1
        else:
            print(f"[SKIP] {prompt_file.relative_to(prompts_dir)} (no change needed)")

    if not args.apply:
        print(f"\n[DRY-RUN] Would update {updated_count} prompt files")
        print("To apply these changes, run with --apply flag")
        return 0 if updated_count == 0 else 1

    print(f"\n✅ Updated {updated_count} prompt files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
