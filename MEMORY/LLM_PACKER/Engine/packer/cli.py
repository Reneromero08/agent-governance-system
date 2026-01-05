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
        "--project-root",
        default="",
        help="Source project root to pack (defaults to this repo root). Output still goes under MEMORY/LLM_PACKER/_packs/ of this repo.",
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
        "--emit-pruned",
        action="store_true",
        help="Generate PRUNED/ output (reduced planning context for LLM navigation).",
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
        help="Write archives: Internal (<pack>/archive/pack.zip) and External (MEMORY/LLM_PACKER/_packs/_archive/<pack_name>.zip).",
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
    parser.add_argument(
        "--p2-runs-dir",
        default="",
        help="Where to write P2 RUN_ROOTS artifacts (defaults to CAPABILITY/RUNS).",
    )
    parser.add_argument(
        "--p2-cas-root",
        default="",
        help="Override CAS root (deterministic tests only).",
    )
    parser.add_argument(
        "--skip-proofs",
        action="store_true",
        help="Skip proof regeneration (Navigation/Proofs/_LATEST).",
    )
    parser.add_argument(
        "--with-proofs",
        action="store_true",
        help="Force proof regeneration even if context implies skipping.",
    )
    args = parser.parse_args()

    project_root = None
    if args.project_root:
        project_root = Path(args.project_root)
        if not project_root.is_absolute():
            project_root = (PROJECT_ROOT / project_root).resolve()
        project_root = project_root.resolve()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    allow_dup = None
    if args.allow_duplicate_hashes:
        allow_dup = True
    elif args.disallow_duplicate_hashes:
        allow_dup = False

    p2_runs_dir = None
    if args.p2_runs_dir:
        p2_runs_dir = Path(args.p2_runs_dir)
        if not p2_runs_dir.is_absolute():
            p2_runs_dir = (PROJECT_ROOT / p2_runs_dir).resolve()
        p2_runs_dir = p2_runs_dir.resolve()

    p2_cas_root = None
    if args.p2_cas_root:
        p2_cas_root = Path(args.p2_cas_root)
        if not p2_cas_root.is_absolute():
            p2_cas_root = (PROJECT_ROOT / p2_cas_root).resolve()
        p2_cas_root = p2_cas_root.resolve()

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
        project_root=project_root,
        p2_runs_dir=p2_runs_dir,
        p2_cas_root=p2_cas_root,
        skip_proofs=args.skip_proofs and not args.with_proofs,
    )
    print(f"Pack created: {pack_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
