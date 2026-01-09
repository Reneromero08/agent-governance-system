from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path for CAPABILITY imports
# Project root is 5 levels up from this file (doc_merge_batch/cli.py -> skill -> utilities -> SKILLS -> CAPABILITY -> PROJECT_ROOT)
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .core import run_job

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

def run_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="doc-merge-batch",
        description="Deterministic diff/merge/verify tool for pairwise, intentional document merging. ALWAYS use verify mode first.",
        epilog="SAFETY: Read README.md before use. Never skip verify mode. Use explicit pairs.json. Write outputs to a dedicated merge directory."
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=["scan","compare","plan","apply","verify","prune_quarantine"],
        help="Operation mode. REQUIRED FIRST STEP: use 'verify' before any destructive actions. Never skip verify."
    )
    ap.add_argument("--root", help="Root directory for scan mode")
    ap.add_argument("--pairs", help="Path to pairs.json (explicit list of file pairs to merge)")
    ap.add_argument(
        "--out",
        default="./MERGE_OUT",
        help="Output directory. RECOMMENDED: Use a dedicated merge directory like _merge/out/ (NOT repo root or working trees)"
    )
    ap.add_argument("--max-file-mb", type=float, default=20, help="Max file size in MB (default: 20)")
    ap.add_argument("--max-pairs", type=int, default=5000, help="Max pairs to process (default: 5000)")
    ap.add_argument("--max-diff-lines", type=int, default=500, help="Max diff lines in report (default: 500)")
    ap.add_argument("--context-lines", type=int, default=3, help="Diff context lines (default: 3)")
    ap.add_argument("--base", choices=["a","b"], default="a", help="Which file to use as merge base (default: a)")
    ap.add_argument(
        "--on-success",
        choices=["none","delete_tracked","quarantine"],
        default="none",
        help="Post-action after successful verify/apply. WARNING: 'delete_tracked' permanently deletes originals if git-tracked+committed+clean. Prefer 'quarantine' for safety. (default: none)"
    )
    ap.add_argument("--ttl-days", type=int, default=14, help="TTL for quarantine mode in days (default: 14)")
    ap.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="DANGER: DATA LOSS RISK. Allow delete_tracked to delete files even if they have uncommitted changes. Use with extreme caution."
    )
    ap.add_argument(
        "--git-commit",
        action="store_true",
        help="Auto-commit merged outputs and deletions after successful verify (only works with delete_tracked mode)"
    )
    ap.add_argument("--git-message", default="housekeeping: merge+prune originals", help="Commit message for --git-commit")
    ap.add_argument("--strip-trailing-ws", action="store_true", help="Strip trailing whitespace during normalization")
    ap.add_argument("--collapse-blank-lines", action="store_true", help="Collapse multiple blank lines during normalization")
    ap.add_argument("--newline", choices=["preserve","lf"], default="lf", help="Newline handling (default: lf)")
    ap.add_argument("--write-report", default=None, help="Write report JSON to this path (defaults to <out>/report.json)")
    args = ap.parse_args(argv)

    payload: Dict[str, Any] = {
        "mode": args.mode,
        "out_dir": args.out,
        "max_file_mb": args.max_file_mb,
        "max_pairs": args.max_pairs,
        "normalization": {
            "newline": args.newline,
            "strip_trailing_ws": bool(args.strip_trailing_ws),
            "collapse_blank_lines": bool(args.collapse_blank_lines),
        },
        "diff": {"max_diff_lines": args.max_diff_lines, "context_lines": args.context_lines},
        "merge": {"base": args.base, "strategy": "append_unique_blocks"},
        "post_actions": {"on_success": args.on_success, "ttl_days": int(args.ttl_days), "require_committed": (not bool(args.allow_uncommitted))},
        "git_commit": {"enabled": bool(args.git_commit), "message": str(args.git_message)},
    }

    if args.mode == "scan":
        if not args.root:
            raise SystemExit("scan mode requires --root")
        payload["root"] = args.root
    elif args.mode == "prune_quarantine":
        # uses --out only
        pass
    else:
        if not args.pairs:
            raise SystemExit("compare/plan/apply/verify require --pairs pairs.json")
        payload["pairs_path"] = args.pairs

    if not GuardedWriter:
        print("Error: GuardedWriter unavailable. Cannot complete CLI job.", file=sys.stderr)
        return 1

    # Heuristic for project root: 5 levels up from CLI script
    # This might be brittle if installed differently.
    # But this tool is part of the repo structure.
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    writer = GuardedWriter(PROJECT_ROOT)
    writer.open_commit_gate() # CLI is allowed to write results

    report = run_job(payload, writer=writer)
    out_dir = Path(args.out)
    writer.mkdir_durable(str(out_dir))
    report_path = Path(args.write_report) if args.write_report else (out_dir / "report.json")
    writer.write_durable(str(report_path), json.dumps(report, indent=2))
    print(report_path.as_posix())
    return 0

if __name__ == "__main__":
    raise SystemExit(run_cli())
