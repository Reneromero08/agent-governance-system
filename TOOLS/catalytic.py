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

from PIPELINES.pipeline_runtime import PipelineRuntime
from PIPELINES.pipeline_verify import verify_pipeline
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
    log_dereference_event,
)


def _resolve_cas_root(*, run_id: str | None, cas_root: str | None) -> Path:
    if cas_root is not None:
        return Path(cas_root)
    if run_id is None:
        raise SystemExit("ERROR: provide --run-id or --cas-root")
    return REPO_ROOT / "CONTRACTS" / "_runs" / run_id / "CAS"


def _resolve_ledger_path(*, run_id: str | None) -> Path | None:
    """Return ledger path if run_id is provided, else None."""
    if run_id is None:
        return None
    return REPO_ROOT / "CONTRACTS" / "_runs" / run_id / "LEDGER.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(prog="catalytic", description="Catalytic CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hash_p = sub.add_parser("hash", help="Expand-by-hash toolbelt")
    hash_p.add_argument("--run-id", default=None, help="Run ID (uses CONTRACTS/_runs/<run_id>/CAS)")
    hash_p.add_argument("--cas-root", default=None, help="CAS root directory (overrides --run-id)")
    hash_p.add_argument("--timestamp", default="CATALYTIC-DPT-LEDGER-SENTINEL", help="Deterministic timestamp for ledger logging")
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

    pipeline_p = sub.add_parser("pipeline", help="Artifact-only pipeline runner")
    pipeline_sub = pipeline_p.add_subparsers(dest="pipe_cmd", required=True)

    run_p = pipeline_sub.add_parser("run", help="Initialize/resume and run pipeline")
    run_p.add_argument("pipeline_id", help="Pipeline ID")
    run_p.add_argument("--spec", required=False, default=None, help="PipelineSpec JSON path (required on first run)")

    status_p = pipeline_sub.add_parser("status", help="Print deterministic pipeline status")
    status_p.add_argument("pipeline_id", help="Pipeline ID")

    verify_p = pipeline_sub.add_parser("verify", help="Fail-closed pipeline verification (artifact-only)")
    verify_p.add_argument("--pipeline-id", required=True, help="Pipeline ID")
    verify_p.add_argument("--runs-root", default="CONTRACTS/_runs", help="Runs root (default: CONTRACTS/_runs)")
    verify_p.add_argument("--strict", action="store_true", help="Enable strict verification (default behavior)")

    args = parser.parse_args()
    store = None
    run_id = None
    timestamp = "CATALYTIC-DPT-LEDGER-SENTINEL"
    ledger_path = None
    if args.cmd == "hash":
        cas_root = _resolve_cas_root(run_id=getattr(args, "run_id", None), cas_root=getattr(args, "cas_root", None))
        store = CatalyticStore(cas_root)
        run_id = getattr(args, "run_id", None)
        timestamp = getattr(args, "timestamp", "CATALYTIC-DPT-LEDGER-SENTINEL")
        ledger_path = _resolve_ledger_path(run_id=run_id)

    try:
        if args.cmd == "pipeline" and args.pipe_cmd == "verify":
            runs_root = Path(args.runs_root)
            if not runs_root.is_absolute():
                runs_root = REPO_ROOT / runs_root
            result = verify_pipeline(
                project_root=REPO_ROOT,
                pipeline_id=args.pipeline_id,
                runs_root=runs_root,
                strict=True,
            )
            if result.get("ok", False):
                details = result.get("details", {})
                steps_verified = details.get("steps_verified", 0)
                sys.stdout.write(f"OK pipeline_id={args.pipeline_id} steps_verified={steps_verified}\n")
                return 0

            details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
            step_id = details.get("step_id")
            run_id = details.get("run_id")
            suffix = ""
            if isinstance(step_id, str) and step_id:
                suffix += f" step_id={step_id}"
            if isinstance(run_id, str) and run_id:
                suffix += f" run_id={run_id}"
            sys.stdout.write(f"FAIL code={result.get('code', 'ERROR')}{suffix}\n")
            return 1

        if args.cmd == "pipeline" and args.pipe_cmd == "status":
            rt = PipelineRuntime(project_root=REPO_ROOT)
            sys.stdout.write(rt.status_text(pipeline_id=args.pipeline_id))
            return 0

        if args.cmd == "pipeline" and args.pipe_cmd == "run":
            rt = PipelineRuntime(project_root=REPO_ROOT)
            spec_path = Path(args.spec) if args.spec is not None else None
            rt.run(pipeline_id=args.pipeline_id, spec_path=spec_path)
            sys.stdout.write(rt.status_text(pipeline_id=args.pipeline_id))
            return 0

        if args.cmd == "hash" and args.hash_cmd == "read":
            out = hash_read_text(
                store=store,
                hash_hex=args.sha256,
                max_bytes=args.max_bytes,
                start=args.start,
                end=args.end,
            )
            sys.stdout.write(out)

            # Log dereference event if run context present
            if ledger_path is not None:
                log_dereference_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    ledger_path=ledger_path,
                    command="read",
                    hash_hex=args.sha256,
                    bounds={"max_bytes": args.max_bytes, "start": args.start, "end": args.end},
                )
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

            # Log dereference event if run context present
            if ledger_path is not None:
                log_dereference_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    ledger_path=ledger_path,
                    command="grep",
                    hash_hex=args.sha256,
                    bounds={"max_bytes": args.max_bytes, "max_matches": args.max_matches, "pattern": args.pattern},
                )
            return 0

        if args.cmd == "hash" and args.hash_cmd == "describe":
            sys.stdout.write(hash_describe(store=store, hash_hex=args.sha256, max_bytes=args.max_bytes) + "\n")

            # Log dereference event if run context present
            if ledger_path is not None:
                log_dereference_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    ledger_path=ledger_path,
                    command="describe",
                    hash_hex=args.sha256,
                    bounds={"max_bytes": args.max_bytes},
                )
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

            # Log dereference event if run context present
            if ledger_path is not None:
                log_dereference_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    ledger_path=ledger_path,
                    command="ast",
                    hash_hex=args.sha256,
                    bounds={"max_bytes": args.max_bytes, "max_nodes": args.max_nodes, "max_depth": args.max_depth},
                )
            return 0

    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    sys.stderr.write("ERROR: unsupported command\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
