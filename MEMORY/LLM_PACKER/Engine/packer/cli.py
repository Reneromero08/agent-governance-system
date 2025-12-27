#!/usr/bin/env python3
"""
CLI entry point for LLM Packer (Phase 1 Modular).
Calls Engine.packer.core.make_pack directly.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .core import make_pack, SCOPE_AGS, SCOPES

def main():
    parser = argparse.ArgumentParser(
        description="Create memory/LLM packs under MEMORY/LLM_PACKER/_packs/."
    )
    parser.add_argument(
        "--scope",
        choices=tuple(sorted(SCOPES.keys())),
        default=SCOPE_AGS.key,
        help="What to pack: default is the full AGS repo.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "delta"),
        default="full",
        help="Pack mode: full includes all included text files.",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "lite"),
        default="full",
        help="Pack profile.",
    )
    parser.add_argument(
        "--split-lite",
        action="store_true",
        help="Generate LITE/ output (renamed from SPLIT_LITE).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate FULL/ output (renamed from COMBINED).",
    )
    parser.add_argument(
        "--stamp",
        default="",
        help="Stamp string for output filenames.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Write a zip archive under MEMORY/LLM_PACKER/_packs/_system/archive/.",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=50 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-entry-bytes",
        type=int,
        default=2 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--allow-duplicate-hashes",
        action="store_true",
    )
    parser.add_argument(
        "--disallow-duplicate-hashes",
        action="store_true",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    allow_dup = None
    if args.allow_duplicate_hashes:
        allow_dup = True
    elif args.disallow_duplicate_hashes:
        allow_dup = False

    pack_dir = make_pack(
        scope_key=args.scope,
        mode=args.mode,
        profile=args.profile,
        split_lite=bool(args.split_lite),
        out_dir=out_dir,
        combined=bool(args.combined),
        stamp=args.stamp or None,
        zip_enabled=bool(args.zip),
        max_total_bytes=int(args.max_total_bytes),
        max_entry_bytes=int(args.max_entry_bytes),
        max_entries=int(args.max_entries),
        allow_duplicate_hashes=allow_dup,
    )
    print(f"Pack created: {pack_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
