#!/usr/bin/env python3
"""
Catalytic Restore Runner CLI (SPECTRUM-06)

Usage:
  # Restore a single run directory into restore_root
  python TOOLS/catalytic_restore.py bundle --run-dir CONTRACTS/_runs/<run_id> --restore-root /abs/path [--json]

  # Restore a chain (explicit order) into restore_root/<run_id>/
  python TOOLS/catalytic_restore.py chain --run-dirs CONTRACTS/_runs/<run1> CONTRACTS/_runs/<run2> --restore-root /abs/path [--json]

Exit codes:
  0: restore succeeded
  1: restore failed
  2: invalid arguments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from CAPABILITY.PRIMITIVES.restore_runner import restore_bundle, restore_chain  # noqa: E402


def _print_json(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="catalytic_restore", add_help=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON result only (no extra text)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bundle = sub.add_parser("bundle", help="Restore a single run directory")
    p_bundle.add_argument("--run-dir", required=True, type=Path)
    p_bundle.add_argument("--restore-root", required=True, type=Path)

    p_chain = sub.add_parser("chain", help="Restore an ordered list of run directories")
    p_chain.add_argument("--run-dirs", required=True, nargs="+", type=Path)
    p_chain.add_argument("--restore-root", required=True, type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "bundle":
        result = restore_bundle(args.run_dir, args.restore_root, strict=True)
    else:
        result = restore_chain(args.run_dirs, args.restore_root, strict=True)

    if args.json:
        _print_json(result)
    else:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True))

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
