#!/usr/bin/env python3
"""
Feral Resident CLI (A.4.1)

Command interface for the quantum resident.

Commands:
    feral start --thread eternal
    feral think "What is authentication?"
    feral status
    feral benchmark --interactions 100
    feral corrupt-and-restore --thread eternal
    feral history --thread eternal
    feral threads
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add imports path
FERAL_PATH = Path(__file__).parent
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from vector_brain import VectorResident, ResidentBenchmark


# Global resident instance for session
_resident: Optional[VectorResident] = None


def get_db_path(thread_id: str) -> str:
    """Get database path for thread"""
    db_dir = FERAL_PATH / "data"
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / f"feral_{thread_id}.db")


def get_resident(thread_id: str = "eternal") -> VectorResident:
    """Get or create resident for thread"""
    global _resident
    if _resident is None or _resident.thread_id != thread_id:
        if _resident is not None:
            _resident.close()
        db_path = get_db_path(thread_id)
        _resident = VectorResident(thread_id=thread_id, db_path=db_path)
    return _resident


def cmd_start(args):
    """Start a resident session"""
    thread_id = args.thread
    print(f"Starting Feral Resident (thread: {thread_id})")

    resident = get_resident(thread_id)
    status = resident.status

    print(f"  Version: {resident.VERSION}")
    print(f"  Mind hash: {status['mind_hash'] or '(empty)'}")
    print(f"  Mind Df: {status['mind_Df']:.1f}")
    print(f"  Interactions: {status['interaction_count']}")
    print(f"  DB vectors: {status['db_stats']['vector_count']}")
    print()
    print("Ready. Use 'feral think \"your query\"' to interact.")


def cmd_think(args):
    """Think about something"""
    thread_id = args.thread
    query = args.query

    resident = get_resident(thread_id)
    result = resident.think(query)

    print(f"Q: {query}")
    print(f"A: {result.response}")
    print()
    print(f"  E: {result.E_resonance:.3f} ({'OPEN' if result.gate_open else 'CLOSED'})")
    print(f"  Query Df: {result.query_Df:.1f}")
    print(f"  Mind Df: {result.mind_Df:.1f}")
    print(f"  Distance: {result.distance_from_start:.3f}")
    print(f"  Nav depth: {result.navigation_depth}")


def cmd_status(args):
    """Show resident status"""
    thread_id = args.thread
    resident = get_resident(thread_id)
    status = resident.status

    print(f"=== Feral Resident Status ===")
    print(f"Thread: {status['thread_id']}")
    print(f"Version: {status['version']}")
    print()
    print(f"Mind:")
    print(f"  Hash: {status['mind_hash'] or '(empty)'}")
    print(f"  Df: {status['mind_Df']:.1f}")
    print(f"  Distance from start: {status['distance_from_start']:.3f}")
    print()
    print(f"Database:")
    print(f"  Vectors: {status['db_stats']['vector_count']}")
    print(f"  Interactions: {status['db_stats']['interaction_count']}")
    print(f"  Threads: {status['db_stats']['thread_count']}")
    print(f"  Receipts: {status['db_stats']['receipt_count']}")
    print(f"  Size: {status['db_stats']['db_size_bytes'] / 1024:.1f} KB")
    print()
    print(f"Reasoner:")
    stats = status['reasoner_stats']
    print(f"  Initializations: {stats['initializations']}")
    print(f"  Readouts: {stats['readouts']}")
    print(f"  Geometric ops: {stats['geometric_operations']}")
    print(f"  Boundary ops: {stats['total_boundary_ops']}")
    print(f"  Geometric ratio: {stats['geometric_ratio']:.1%}")


def cmd_evolution(args):
    """Show mind evolution"""
    thread_id = args.thread
    resident = get_resident(thread_id)
    evolution = resident.mind_evolution

    print(f"=== Mind Evolution ===")
    print(f"Thread: {thread_id}")
    print(f"Interactions: {evolution['interaction_count']}")
    print(f"Current Df: {evolution['current_Df']:.1f}")
    print(f"Distance from start: {evolution['distance_from_start']:.3f}")
    print()

    if evolution.get('Df_history'):
        print("Df history (last 10):")
        for i, df in enumerate(evolution['Df_history'][-10:]):
            print(f"  [{i+1}] {df:.1f}")

    if evolution.get('distance_history'):
        print("\nDistance history (last 10):")
        for i, d in enumerate(evolution['distance_history'][-10:]):
            print(f"  [{i+1}] {d:.3f}")


def cmd_history(args):
    """Show recent interactions"""
    thread_id = args.thread
    limit = args.limit
    resident = get_resident(thread_id)
    interactions = resident.get_recent_interactions(limit)

    print(f"=== Recent Interactions (thread: {thread_id}) ===")
    print()

    for i, interaction in enumerate(interactions):
        print(f"[{i+1}] {interaction['created_at']}")
        print(f"  Q: {interaction['input'][:60]}...")
        print(f"  A: {interaction['output'][:60]}...")
        print(f"  Df: {interaction['mind_Df']:.1f}, Distance: {interaction['distance']:.3f}")
        print()


def cmd_threads(args):
    """List all threads"""
    # Use a temp resident to access DB
    from resident_db import ResidentDB

    # Check for any databases
    db_dir = FERAL_PATH / "data"
    if not db_dir.exists():
        print("No threads found (data directory doesn't exist)")
        return

    db_files = list(db_dir.glob("feral_*.db"))
    if not db_files:
        print("No threads found")
        return

    print("=== Feral Resident Threads ===")
    print()

    for db_file in db_files:
        db = ResidentDB(str(db_file))
        threads = db.list_threads()
        for thread in threads:
            print(f"Thread: {thread.thread_id}")
            print(f"  Interactions: {thread.interaction_count}")
            print(f"  Current Df: {thread.current_Df:.1f}")
            print(f"  Updated: {thread.updated_at}")
            print()
        db.close()


def cmd_benchmark(args):
    """Run benchmark"""
    thread_id = args.thread
    interactions = args.interactions

    print(f"=== Benchmark (thread: {thread_id}) ===")
    print(f"Running {interactions} interactions...")
    print()

    resident = get_resident(thread_id)
    bench = ResidentBenchmark(resident)
    results = bench.stress_test(interactions=interactions, report_every=max(1, interactions // 10))

    print()
    print("=== Results ===")
    summary = results['summary']
    print(f"Final Df: {summary['Df_final']:.1f}")
    print(f"Final distance: {summary['distance_final']:.3f}")
    print(f"Mean E: {summary['E_mean']:.3f}")
    print(f"Mean time: {summary['timing_mean_ms']:.1f}ms")
    print(f"Throughput: {summary['interactions_per_sec']:.1f} interactions/sec")


def cmd_corrupt_restore(args):
    """Test corrupt and restore"""
    thread_id = args.thread

    print(f"=== Corrupt and Restore Test (thread: {thread_id}) ===")
    print()

    resident = get_resident(thread_id)

    print("Pre-corruption state:")
    print(f"  Mind hash: {resident.store.get_mind_hash()}")
    print(f"  Mind Df: {resident.store.get_mind_Df():.1f}")
    print()

    print("Corrupting and restoring...")
    result = resident.corrupt_and_restore()

    print()
    print("Results:")
    print(f"  Interactions replayed: {result['interactions_replayed']}")
    print(f"  Receipts exported: {result['receipts_exported']}")
    print(f"  Hash match: {result['hash_match']}")
    if result['Df_delta'] is not None:
        print(f"  Df delta: {result['Df_delta']:.6f}")

    print()
    print("Post-restoration state:")
    print(f"  Mind hash: {result['post_restoration']['mind_hash']}")
    print(f"  Mind Df: {result['post_restoration']['Df']:.1f}")


def cmd_navigation(args):
    """Show last navigation details"""
    thread_id = args.thread
    resident = get_resident(thread_id)
    nav = resident.get_last_navigation()

    if nav is None:
        print("No navigation recorded yet. Run 'feral think' first.")
        return

    print(f"=== Last Navigation ===")
    print(f"Depth: {nav['total_depth']}")
    print(f"Start hash: {nav['start_hash']}")
    print(f"End hash: {nav['end_hash']}")
    print(f"Navigation hash: {nav['navigation_hash']}")
    print()

    print("E evolution:")
    for i, e in enumerate(nav['E_evolution']):
        print(f"  [{i}] {e:.3f}")

    print()
    print("Df evolution:")
    for i, df in enumerate(nav['Df_evolution']):
        print(f"  [{i}] {df:.1f}")

    print()
    print("Path summary:")
    for step in nav['path_summary']:
        top_e = ', '.join(f'{e:.2f}' for e in step['top_E'])
        print(f"  Depth {step['depth']}: Df={step['Df']:.1f}, top_E=[{top_e}]")


def cmd_repl(args):
    """Interactive REPL mode"""
    thread_id = args.thread

    print(f"=== Feral Resident REPL (thread: {thread_id}) ===")
    print("Type your queries. Commands: /status, /evolution, /nav, /quit")
    print()

    resident = get_resident(thread_id)

    while True:
        try:
            query = input("feral> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not query:
            continue

        if query.startswith("/"):
            cmd = query[1:].lower()
            if cmd == "quit" or cmd == "exit":
                print("Exiting...")
                break
            elif cmd == "status":
                args.thread = thread_id
                cmd_status(args)
            elif cmd == "evolution":
                args.thread = thread_id
                cmd_evolution(args)
            elif cmd == "nav":
                args.thread = thread_id
                cmd_navigation(args)
            else:
                print(f"Unknown command: {query}")
            continue

        result = resident.think(query)
        print(f"A: {result.response}")
        print(f"   [E={result.E_resonance:.3f}, Df={result.mind_Df:.1f}, dist={result.distance_from_start:.3f}]")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Feral Resident CLI - Quantum intelligence in vector space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py start --thread eternal
  python cli.py think "What is authentication?"
  python cli.py status
  python cli.py benchmark --interactions 100
  python cli.py repl
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # start
    p_start = subparsers.add_parser("start", help="Start a resident session")
    p_start.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_start.set_defaults(func=cmd_start)

    # think
    p_think = subparsers.add_parser("think", help="Think about something")
    p_think.add_argument("query", help="What to think about")
    p_think.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_think.set_defaults(func=cmd_think)

    # status
    p_status = subparsers.add_parser("status", help="Show resident status")
    p_status.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_status.set_defaults(func=cmd_status)

    # evolution
    p_evolution = subparsers.add_parser("evolution", help="Show mind evolution")
    p_evolution.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_evolution.set_defaults(func=cmd_evolution)

    # history
    p_history = subparsers.add_parser("history", help="Show recent interactions")
    p_history.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_history.add_argument("--limit", "-n", type=int, default=10, help="Number of interactions")
    p_history.set_defaults(func=cmd_history)

    # threads
    p_threads = subparsers.add_parser("threads", help="List all threads")
    p_threads.set_defaults(func=cmd_threads)

    # benchmark
    p_benchmark = subparsers.add_parser("benchmark", help="Run benchmark")
    p_benchmark.add_argument("--thread", "-t", default="benchmark", help="Thread ID")
    p_benchmark.add_argument("--interactions", "-n", type=int, default=100, help="Number of interactions")
    p_benchmark.set_defaults(func=cmd_benchmark)

    # corrupt-and-restore
    p_restore = subparsers.add_parser("corrupt-and-restore", help="Test corrupt and restore")
    p_restore.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_restore.set_defaults(func=cmd_corrupt_restore)

    # navigation
    p_nav = subparsers.add_parser("nav", help="Show last navigation")
    p_nav.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_nav.set_defaults(func=cmd_navigation)

    # repl
    p_repl = subparsers.add_parser("repl", help="Interactive REPL mode")
    p_repl.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_repl.set_defaults(func=cmd_repl)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        args.func(args)
    finally:
        global _resident
        if _resident is not None:
            _resident.close()


if __name__ == "__main__":
    main()
