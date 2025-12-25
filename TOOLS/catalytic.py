#!/usr/bin/env python3

"""
Unified Catalytic CLI.

This ticket implements the expand-by-hash toolbelt:
  catalytic hash read|grep|describe|ast

Storage location:
- Commands operate on CAS objects by hash.
- Provide either `--run-id` (uses `CONTRACTS/_runs/<run_id>/CAS`) or `--cas-root`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.cas_store import CatalyticStore
from PRIMITIVES.hash_toolbelt import (
    DEFAULT_AST_MAX_BYTES,
    DEFAULT_AST_MAX_DEPTH,
    DEFAULT_AST_MAX_NODES,
    DEFAULT_DESCRIBE_MAX_BYTES,
    DEFAULT_GREP_MAX_BYTES,
    DEFAULT_GREP_MAX_MATCHES,
    DEFAULT_READ_MAX_BYTES,
    hash_ast,
    hash_describe,
    hash_grep,
    hash_read_text,
)


def _resolve_cas_root(*, run_id: str | None, cas_root: str | None) -> Path:
    if cas_root is not None:
        return Path(cas_root)
    if run_id is None:
        raise SystemExit("ERROR: provide --run-id or --cas-root")
    return REPO_ROOT / "CONTRACTS" / "_runs" / run_id / "CAS"


def main() -> int:
    parser = argparse.ArgumentParser(prog="catalytic", description="Catalytic CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hash_p = sub.add_parser("hash", help="Expand-by-hash toolbelt")
    hash_p.add_argument("--run-id", default=None, help="Run ID (uses CONTRACTS/_runs/<run_id>/CAS)")
    hash_p.add_argument("--cas-root", default=None, help="CAS root directory (overrides --run-id)")
    hash_sub = hash_p.add_subparsers(dest="hash_cmd", required=True)

    read_p = hash_sub.add_parser("read", help="Read bounded bytes from a CAS object")
    read_p.add_argument("sha256", help="SHA-256 hex (lowercase, 64 chars)")
    read_p.add_argument("--max-bytes", type=int, default=DEFAULT_READ_MAX_BYTES)
    read_p.add_argument("--start", type=int, default=0)
    read_p.add_argument("--end", type=int, default=None)

    grep_p = hash_sub.add_parser("grep", help="Bounded substring search within a CAS object")
    grep_p.add_argument("sha256", help="SHA-256 hex (lowercase, 64 chars)")
    grep_p.add_argument("pattern", help="Plain substring (UTF-8)")
    grep_p.add_argument("--max-bytes", type=int, default=DEFAULT_GREP_MAX_BYTES)
    grep_p.add_argument("--max-matches", type=int, default=DEFAULT_GREP_MAX_MATCHES)

    describe_p = hash_sub.add_parser("describe", help="Bounded content description for a CAS object")
    describe_p.add_argument("sha256", help="SHA-256 hex (lowercase, 64 chars)")
    describe_p.add_argument("--max-bytes", type=int, default=DEFAULT_DESCRIBE_MAX_BYTES)

    ast_p = hash_sub.add_parser("ast", help="Bounded AST outline (Python only)")
    ast_p.add_argument("sha256", help="SHA-256 hex (lowercase, 64 chars)")
    ast_p.add_argument("--max-bytes", type=int, default=DEFAULT_AST_MAX_BYTES)
    ast_p.add_argument("--max-nodes", type=int, default=DEFAULT_AST_MAX_NODES)
    ast_p.add_argument("--max-depth", type=int, default=DEFAULT_AST_MAX_DEPTH)

    args = parser.parse_args()

    cas_root = _resolve_cas_root(run_id=getattr(args, "run_id", None), cas_root=getattr(args, "cas_root", None))
    store = CatalyticStore(cas_root)

    try:
        if args.cmd == "hash" and args.hash_cmd == "read":
            out = hash_read_text(
                store=store,
                hash_hex=args.sha256,
                max_bytes=args.max_bytes,
                start=args.start,
                end=args.end,
            )
            sys.stdout.write(out)
            return 0

        if args.cmd == "hash" and args.hash_cmd == "grep":
            matches = hash_grep(
                store=store,
                hash_hex=args.sha256,
                pattern=args.pattern,
                max_bytes=args.max_bytes,
                max_matches=args.max_matches,
            )
            for m in matches:
                sys.stdout.write(m.format() + "\n")
            return 0

        if args.cmd == "hash" and args.hash_cmd == "describe":
            sys.stdout.write(hash_describe(store=store, hash_hex=args.sha256, max_bytes=args.max_bytes) + "\n")
            return 0

        if args.cmd == "hash" and args.hash_cmd == "ast":
            sys.stdout.write(
                hash_ast(
                    store=store,
                    hash_hex=args.sha256,
                    max_bytes=args.max_bytes,
                    max_nodes=args.max_nodes,
                    max_depth=args.max_depth,
                )
                + "\n"
            )
            return 0

    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    sys.stderr.write("ERROR: unsupported command\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

