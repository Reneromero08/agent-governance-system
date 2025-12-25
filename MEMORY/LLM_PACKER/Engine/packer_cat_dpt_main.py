#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from packer import PROJECT_ROOT, make_pack


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a CATALYTIC-DPT MAIN pack (excludes CATALYTIC-DPT/LAB) under MEMORY/LLM_PACKER/_packs/."
    )
    parser.add_argument("--mode", choices=("full", "delta"), default="full")
    parser.add_argument("--profile", choices=("full", "lite"), default="full")
    parser.add_argument("--split-lite", action="store_true")
    parser.add_argument("--out-dir", default="", help="Output directory (must be under MEMORY/LLM_PACKER/_packs/).")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--stamp", default="", help="Stamp for combined output filenames.")
    parser.add_argument("--zip", action="store_true", help="Write a zip archive under MEMORY/LLM_PACKER/_packs/_system/archive/.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    pack_dir = make_pack(
        scope_key="catalytic-dpt",
        mode=args.mode,
        profile=args.profile,
        split_lite=bool(args.split_lite),
        out_dir=out_dir,
        combined=bool(args.combined),
        stamp=args.stamp or None,
        zip_enabled=bool(args.zip),
    )
    print(f"Pack created: {pack_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
