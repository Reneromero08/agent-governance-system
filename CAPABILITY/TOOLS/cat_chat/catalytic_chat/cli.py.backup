#!/usr/bin/env python3
"""
Catalytic Chat CLI

Command-line interface for building and querying the section index.

Roadmap Phase: Phase 1 — Substrate + deterministic indexing
"""

import sys
import argparse
import json
from pathlib import Path

from catalytic_chat.section_extractor import extract_sections
from catalytic_chat.section_indexer import SectionIndexer, build_index
from catalytic_chat.symbol_registry import SymbolRegistry, SymbolError
from catalytic_chat.symbol_resolver import SymbolResolver, ResolverError, resolve_symbol
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError


def cmd_build(args) -> int:
    """Build section index.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    indexer = SectionIndexer(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        sections = extract_sections(file_path, args.repo_root)
        print(f"Extracted {len(sections)} sections from {file_path}\n")

        for i, section in enumerate(sections, 1):
            print(f"[{i}] {section.section_id[:16]}...")
            print(f"    Heading: {' > '.join(section.heading_path)}")
            print(f"    Lines: {section.line_start}-{section.line_end}")
            print(f"    Hash: {section.content_hash[:16]}...")
            print()

        return 0
    except Exception as e:
        print(f"[FAIL] Extraction failed: {e}")
        return 1


def cmd_verify(args) -> int:
    """Verify index determinism.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    indexer = SectionIndexer(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        success = indexer.verify_determinism()
        return 0 if success else 1
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        return 1


def cmd_get(args) -> int:
    """Get section by ID with optional slice.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .section_indexer import SectionIndexer

    indexer = SectionIndexer(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        section_id = getattr(args, 'slice', None)
        content, content_hash, applied_slice, lines_applied, chars_applied = \
            indexer.get_section_content(args.section_id, section_id)

        print(content, end='')
        sys.stderr.write(f"section_id: {args.section_id}\n")
        sys.stderr.write(f"slice: {applied_slice}\n")
        sys.stderr.write(f"content_hash: {content_hash[:16]}...\n")
        sys.stderr.write(f"lines_applied: {lines_applied}\n")
        sys.stderr.write(f"chars_applied: {chars_applied}\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"[FAIL] Failed to get section: {e}\n")
        return 1


def cmd_extract(args) -> int:
    """Extract sections from a file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"[FAIL] File not found: {file_path}")
        return 1

    try:
        sections = extract_sections(file_path, args.repo_root)
        print(f"Extracted {len(sections)} sections from {file_path}\n")

        for i, section in enumerate(sections, 1):
            print(f"[{i}] {section.section_id[:16]}...")
            print(f"    Heading: {' > '.join(section.heading_path)}")
            print(f"    Lines: {section.line_start}-{section.line_end}")
            print(f"    Hash: {section.content_hash[:16]}...")
            print()

        return 0
    except Exception as e:
        print(f"[FAIL] Extraction failed: {e}")
        return 1


def cmd_symbols_add(args) -> int:
    """Add symbol to registry.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry, SymbolError

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        timestamp = registry.add_symbol(
            symbol_id=args.symbol_id,
            target_ref=args.section,
            default_slice=args.default_slice
        )
        print(f"[OK] Symbol added: {args.symbol_id}")
        print(f"      Target: {args.section}")
        if args.default_slice:
            print(f"      Default slice: {args.default_slice}")
        print(f"      Created: {timestamp}")
        return 0
    except SymbolError as e:
        print(f"[FAIL] {e}")
        return 1
    except Exception as e:
        print(f"[FAIL] Failed to add symbol: {e}")
        return 1


def cmd_symbols_get(args) -> int:
    """Get symbol from registry.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry, Symbol

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        symbol = registry.get_symbol(args.symbol_id)

        if symbol is None:
            print(f"[FAIL] Symbol not found: {args.symbol_id}")
            return 1

        print(f"Symbol: {symbol.symbol_id}")
        print(f"  Target Type: {symbol.target_type}")
        print(f"  Target Ref: {symbol.target_ref}")
        if symbol.default_slice:
            print(f"  Default Slice: {symbol.default_slice}")
        print(f"  Created: {symbol.created_at}")
        print(f"  Updated: {symbol.updated_at}")
        return 0
    except Exception as e:
        print(f"[FAIL] Failed to get symbol: {e}")
        return 1


def cmd_symbols_list(args) -> int:
    """List symbols from registry.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        prefix = getattr(args, 'prefix', None)
        symbols = registry.list_symbols(prefix)

        print(f"Listing {len(symbols)} symbols")
        if prefix:
            print(f"  Prefix: {prefix}")
        print()

        for symbol in symbols:
            print(f"  {symbol.symbol_id}")
            print(f"    Target: {symbol.target_ref}")
            if symbol.default_slice:
                print(f"    Slice: {symbol.default_slice}")
        return 0
    except Exception as e:
        print(f"[FAIL] Failed to list symbols: {e}")
        return 1


def cmd_symbols_verify(args) -> int:
    """Verify symbol registry integrity.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        success = registry.verify()
        return 0 if success else 1
    except Exception as e:
        print(f"[FAIL] Verification error: {e}")
        return 1


def cmd_resolve(args) -> int:
    """Resolve symbol to content with caching.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_resolver import ResolverError
    from .symbol_registry import SymbolRegistry

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    resolver = SymbolResolver(
        repo_root=args.repo_root,
        substrate_mode=args.substrate,
        symbol_registry=registry
    )

    try:
        payload, cache_hit = resolver.resolve(
            symbol_id=args.symbol_id,
            slice_expr=args.slice,
            run_id=args.run_id
        )

        print(payload, end='')
        sys.stderr.write(f"[CACHE {'HIT' if cache_hit else 'MISS'}]\n")
        return 0
    except ResolverError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"[FAIL] Resolution error: {e}\n")
        return 1


def cmd_cassette_verify(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        cassette.verify_cassette(getattr(args, 'run_id', None))
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_cassette_post(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        with open(args.json, 'r') as f:
            payload = json.load(f)
        
        message_id, job_id = cassette.post_message(
            payload=payload,
            run_id=args.run_id,
            source=args.source,
            idempotency_key=args.idempotency_key
        )
        
        print(f"[OK] Message posted")
        print(f"      message_id: {message_id}")
        print(f"      job_id: {job_id}")
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except FileNotFoundError:
        sys.stderr.write(f"[FAIL] File not found: {args.json}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[FAIL] Invalid JSON: {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_cassette_claim(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        result = cassette.claim_step(
            run_id=args.run_id,
            worker_id=args.worker,
            ttl_seconds=args.ttl
        )
        
        print(f"[OK] Step claimed")
        print(f"      step_id: {result['step_id']}")
        print(f"      job_id: {result['job_id']}")
        print(f"      message_id: {result['message_id']}")
        print(f"      ordinal: {result['ordinal']}")
        print(f"      fencing_token: {result['fencing_token']}")
        print(f"      lease_expires_at: {result['lease_expires_at']}")
        print()
        print("Payload:")
        print(json.dumps(result['payload'], indent=2))
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_cassette_complete(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        with open(args.receipt, 'r') as f:
            receipt_payload = json.load(f)
        
        receipt_id = cassette.complete_step(
            run_id=args.run_id,
            step_id=args.step,
            worker_id=args.worker,
            fencing_token=args.token,
            receipt_payload=receipt_payload,
            outcome=args.outcome
        )
        
        print(f"[OK] Step completed")
        print(f"      receipt_id: {receipt_id}")
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except FileNotFoundError:
        sys.stderr.write(f"[FAIL] File not found: {args.receipt}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[FAIL] Invalid JSON: {e}\n")
        return 1
    finally:
        cassette.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Catalytic Chat CLI",
        epilog="Roadmap Phase: Phase 1 — Substrate + deterministic indexing"
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root path (default: current working directory)"
    )
    parser.add_argument(
        "--substrate",
        choices=["sqlite", "jsonl"],
        default="sqlite",
        help="Substrate mode (default: sqlite)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    build_parser = subparsers.add_parser("build", help="Build section index")
    build_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Build incrementally (only changed files)"
    )

    verify_parser = subparsers.add_parser("verify", help="Verify index determinism")

    get_parser = subparsers.add_parser("get", help="Get section by ID")
    get_parser.add_argument("section_id", help="Section ID")
    get_parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice expression (e.g., lines[0:100], chars[0:500], head(50), tail(20))"
    )

    extract_parser = subparsers.add_parser("extract", help="Extract sections from file")
    extract_parser.add_argument("file_path", help="Path to file")

    symbols_parser = subparsers.add_parser("symbols", help="Symbol registry commands")
    symbols_subparsers = symbols_parser.add_subparsers(dest="symbols_command", help="Symbol commands")

    symbols_add_parser = symbols_subparsers.add_parser("add", help="Add symbol to registry")
    symbols_add_parser.add_argument("symbol_id", help="Symbol ID (must start with @)")
    symbols_add_parser.add_argument("--section", required=True, help="Section ID to reference")
    symbols_add_parser.add_argument("--default-slice", help="Default slice expression")

    symbols_get_parser = symbols_subparsers.add_parser("get", help="Get symbol from registry")
    symbols_get_parser.add_argument("symbol_id", help="Symbol ID")

    symbols_list_parser = symbols_subparsers.add_parser("list", help="List symbols")
    symbols_list_parser.add_argument("--prefix", help="Filter by prefix (e.g., @CANON/)")

    symbols_verify_parser = symbols_subparsers.add_parser("verify", help="Verify symbol registry")

    resolve_parser = subparsers.add_parser("resolve", help="Resolve symbol to content with caching")

    cassette_parser = subparsers.add_parser("cassette", help="Message cassette commands (Phase 3)")
    cassette_subparsers = cassette_parser.add_subparsers(dest="cassette_command", help="Cassette commands")

    cassette_verify_parser = cassette_subparsers.add_parser("verify", help="Verify cassette integrity")
    cassette_verify_parser.add_argument("--run-id", type=str, default=None, help="Verify specific run")

    cassette_post_parser = cassette_subparsers.add_parser("post", help="Post message to cassette")
    cassette_post_parser.add_argument("--json", type=Path, required=True, help="JSON file with message payload")
    cassette_post_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    cassette_post_parser.add_argument("--source", type=str, required=True, 
                                    choices=["USER", "PLANNER", "SYSTEM", "WORKER"], help="Message source")
    cassette_post_parser.add_argument("--idempotency-key", type=str, default=None, help="Idempotency key")

    cassette_claim_parser = cassette_subparsers.add_parser("claim", help="Claim a pending step")
    cassette_claim_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    cassette_claim_parser.add_argument("--worker", type=str, required=True, help="Worker ID")
    cassette_claim_parser.add_argument("--ttl", type=int, default=300, help="TTL in seconds (default: 300)")

    cassette_complete_parser = cassette_subparsers.add_parser("complete", help="Complete a step")
    cassette_complete_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    cassette_complete_parser.add_argument("--step", type=str, required=True, help="Step ID")
    cassette_complete_parser.add_argument("--worker", type=str, required=True, help="Worker ID")
    cassette_complete_parser.add_argument("--token", type=int, required=True, help="Fencing token")
    cassette_complete_parser.add_argument("--receipt", type=Path, required=True, help="JSON file with receipt payload")
    cassette_complete_parser.add_argument("--outcome", type=str, required=True,
                                        choices=["SUCCESS", "FAILURE", "ABORTED"], help="Outcome")
    resolve_parser.add_argument("symbol_id", help="Symbol ID")
    resolve_parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice expression (e.g., lines[0:100], chars[0:500], head(50), tail(20))"
    )
    resolve_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for caching"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "build": cmd_build,
        "verify": cmd_verify,
        "get": cmd_get,
        "extract": cmd_extract,
        "symbols": cmd_symbols_add,
        "resolve": cmd_resolve,
        "cassette": cmd_cassette_verify
    }

    if args.command not in commands:
        print(f"[FAIL] Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    if args.command == "symbols":
        symbols_commands = {
            "add": cmd_symbols_add,
            "get": cmd_symbols_get,
            "list": cmd_symbols_list,
            "verify": cmd_symbols_verify
        }

        if args.symbols_command not in symbols_commands:
            print(f"[FAIL] Unknown symbols command: {args.symbols_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(commands[args.command](args))

    if args.command == "cassette":
        cassette_commands = {
            "verify": cmd_cassette_verify,
            "post": cmd_cassette_post,
            "claim": cmd_cassette_claim,
            "complete": cmd_cassette_complete
        }

        if args.cassette_command not in cassette_commands:
            print(f"[FAIL] Unknown cassette command: {args.cassette_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(cassette_commands[args.cassette_command](args))


if __name__ == '__main__':
    main()
