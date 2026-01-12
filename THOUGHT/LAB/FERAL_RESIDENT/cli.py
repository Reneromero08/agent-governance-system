#!/usr/bin/env python3
"""
Feral Resident CLI (A.4.1 + B.3)

Command interface for the quantum resident.

Commands:
    feral start --thread eternal
    feral think "What is authentication?"
    feral status
    feral benchmark --interactions 100
    feral corrupt-and-restore --thread eternal
    feral history --thread eternal
    feral threads

Symbol Evolution (B.3):
    feral symbol-evolution --thread eternal
    feral notations --thread eternal
    feral breakthroughs --thread eternal
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
    print(f"  E_resonance: {result.E_resonance:.3f} ({'OPEN' if result.gate_open else 'CLOSED'})")
    print(f"  E_compression: {result.E_compression:.3f} (B.3: output resonance)")
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


# === Emergence Tracking (B.2) ===

def cmd_metrics(args):
    """Show emergence metrics (B.2)"""
    from emergence import detect_protocols, print_emergence_report
    import json as json_module

    thread_id = args.thread

    if args.json:
        metrics = detect_protocols(thread_id)
        print(json_module.dumps(metrics, indent=2, default=str))
    else:
        print_emergence_report(thread_id)


# === Symbol Evolution (B.3) ===

def cmd_symbol_evolution(args):
    """Show symbol evolution dashboard (B.3)"""
    from symbol_evolution import SymbolEvolutionTracker
    import json as json_module

    thread_id = args.thread
    tracker = SymbolEvolutionTracker(thread_id)

    if args.json:
        report = tracker.get_evolution_report()
        print(json_module.dumps(report, indent=2, default=str))
    else:
        tracker.print_evolution_dashboard()


def cmd_notations(args):
    """Show notation registry (B.3)"""
    from symbol_evolution import NotationRegistry

    thread_id = args.thread
    registry = NotationRegistry(thread_id)

    print(f"=== Notation Registry (B.3) ===")
    print(f"Thread: {thread_id}")
    print()

    if not registry.registry:
        print("No notations registered yet.")
        print("Run interactions to detect emerging patterns.")
        return

    # Sort by frequency
    sorted_notations = sorted(
        registry.registry.values(),
        key=lambda x: x.frequency,
        reverse=True
    )

    print(f"Total patterns: {len(sorted_notations)}")
    print(f"Active (freq>=5): {len(registry.get_active_notations())}")
    print()

    for entry in sorted_notations[:args.limit]:
        print(f"Pattern: '{entry.pattern}'")
        print(f"  Type: {entry.pattern_type}")
        print(f"  Frequency: {entry.frequency}")
        print(f"  First seen: {entry.first_seen}")
        print(f"  First session: {entry.first_session}")
        if entry.contexts:
            print(f"  Example context: {entry.contexts[0][:80]}...")
        if entry.meaning_inferred:
            print(f"  Meaning: {entry.meaning_inferred}")
        print()


def cmd_breakthroughs(args):
    """Show breakthrough sessions (B.3)"""
    from symbol_evolution import PointerRatioTracker

    thread_id = args.thread
    tracker = PointerRatioTracker(thread_id)

    print(f"=== Breakthrough Sessions (B.3) ===")
    print(f"Thread: {thread_id}")
    print()

    breakthroughs = tracker.get_breakthroughs()

    if not breakthroughs:
        print("No breakthroughs detected yet.")
        print("A breakthrough = pointer_ratio jump > 0.1 in a single session.")
        return

    print(f"Total breakthroughs: {len(breakthroughs)}")
    print()

    for b in breakthroughs:
        print(f"Session: {b.session_id}")
        print(f"  Timestamp: {b.timestamp}")
        print(f"  Pointer ratio: {b.pointer_ratio:.4f}")
        print(f"  Delta: +{b.delta_from_previous:.4f}")
        print(f"  Symbols: {b.symbols_count}, Hashes: {b.hashes_count}")
        print(f"  Rolling avg (10): {b.rolling_average_10:.4f}")
        print()

    # Show trend
    trend = tracker.get_trend()
    print(f"Overall trend: {trend['trend']}")
    print(f"Current ratio: {trend['current_ratio']:.4f}")
    print(f"Goal: {trend['goal']} ({trend['progress']*100:.1f}% progress)")


# === Paper Management Commands (B.1) ===

def cmd_papers_register(args):
    """Register a paper for indexing"""
    from paper_indexer import PaperIndexer
    indexer = PaperIndexer()

    paper = indexer.register_paper(
        arxiv_id=args.arxiv,
        short_name=args.name,
        title=args.title or f"Paper {args.arxiv}",
        category=args.category,
        pdf_path=args.pdf
    )

    print(f"Registered: {paper.primary_symbol}")
    print(f"  Alias: {paper.alias_symbol}")
    print(f"  Title: {paper.title}")
    print(f"  Category: {paper.category}")
    if paper.pdf_path:
        print(f"  PDF: {paper.pdf_path}")


def cmd_papers_convert(args):
    """Set markdown path after PDF conversion"""
    from paper_indexer import PaperIndexer
    indexer = PaperIndexer()

    paper = indexer.set_markdown_path(args.arxiv, args.markdown)

    print(f"Set markdown path for @Paper-{args.arxiv}")
    print(f"  Markdown: {paper['markdown_path']}")
    print(f"  Status: {paper['status']}")
    print()
    print("Now run: feral papers index --arxiv " + args.arxiv)


def cmd_papers_index(args):
    """Index a paper (or all papers)"""
    from paper_indexer import PaperIndexer
    indexer = PaperIndexer()

    if args.all:
        # Index all converted papers
        indexed = 0
        for arxiv_id, paper in indexer.manifest["papers"].items():
            if paper["status"] == "converted":
                try:
                    result = indexer.index_paper(arxiv_id)
                    print(f"Indexed @Paper-{arxiv_id}: {len(result['chunks'])} chunks")
                    indexed += 1
                except Exception as e:
                    print(f"Failed @Paper-{arxiv_id}: {e}")
        print(f"\nTotal indexed: {indexed}")
    else:
        if not args.arxiv:
            print("Error: --arxiv required (or use --all)")
            return

        result = indexer.index_paper(args.arxiv)
        print(f"Indexed @Paper-{args.arxiv}")
        print(f"  Chunks: {len(result['chunks'])}")
        if result['Df_values']:
            print(f"  Df range: {min(result['Df_values']):.1f} - {max(result['Df_values']):.1f}")


def cmd_papers_list(args):
    """List registered papers"""
    from paper_indexer import PaperIndexer
    indexer = PaperIndexer()

    papers = indexer.list_papers(category=args.category, status=args.status)

    print("=== Paper Corpus ===")
    if not papers:
        print("No papers found.")
        return

    for paper in papers:
        status_icon = {"registered": "o", "converted": "-", "indexed": "*"}.get(
            paper["status"], "?"
        )
        print(f"[{status_icon}] {paper['primary_symbol']} ({paper['alias_symbol']})")
        print(f"    {paper.get('title', 'No title')}")
        print(f"    Category: {paper['category']}, Status: {paper['status']}")


def cmd_papers_query(args):
    """Query papers using E (Born rule)"""
    from paper_indexer import PaperIndexer
    indexer = PaperIndexer()

    results = indexer.query_papers(args.query, k=args.k)

    print(f"=== Query: '{args.query}' ===")
    if not results:
        print("No indexed papers to search.")
        return

    for i, r in enumerate(results):
        print(f"[{i+1}] {r['paper']} - {r['heading']}")
        print(f"    E={r['E']:.3f}, Df={r['Df']:.1f}")
        if args.preview:
            print(f"    Preview: {r['content_preview'][:100]}...")


def cmd_papers_status(args):
    """Show paper corpus status"""
    from paper_indexer import PaperIndexer
    indexer = PaperIndexer()

    stats = indexer.get_stats()

    print("=== Paper Corpus Status ===")
    print(f"Total registered: {stats['total']}")
    print(f"Indexed: {stats['indexed']}")
    print()
    print("By status:")
    for status, count in stats.get("by_status", {}).items():
        print(f"  {status}: {count}")
    print()
    print("By category:")
    for cat, count in stats.get("by_category", {}).items():
        print(f"  {cat}: {count}")
    print()
    if stats.get("Df_mean"):
        print(f"Df statistics:")
        print(f"  Min: {stats['Df_min']:.1f}")
        print(f"  Max: {stats['Df_max']:.1f}")
        print(f"  Mean: {stats['Df_mean']:.1f}")


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

Symbol evolution (B.3):
  python cli.py symbol-evolution --thread eternal
  python cli.py notations --thread eternal --limit 10
  python cli.py breakthroughs --thread eternal

Paper management (B.1):
  python cli.py papers register --arxiv 2310.06816 --name Vec2Text --category vec2text
  python cli.py papers convert --arxiv 2310.06816 --markdown markdown/@Paper-2310.06816.md
  python cli.py papers index --arxiv 2310.06816
  python cli.py papers query "embedding inversion" --k 5
  python cli.py papers status
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

    # metrics (B.2)
    p_metrics = subparsers.add_parser("metrics", help="Show emergence metrics (B.2)")
    p_metrics.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_metrics.add_argument("--json", action="store_true", help="Output as JSON")
    p_metrics.set_defaults(func=cmd_metrics)

    # === Symbol Evolution (B.3) ===

    # symbol-evolution
    p_symevo = subparsers.add_parser("symbol-evolution", help="Show symbol evolution dashboard (B.3)")
    p_symevo.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_symevo.add_argument("--json", action="store_true", help="Output as JSON")
    p_symevo.set_defaults(func=cmd_symbol_evolution)

    # notations
    p_notations = subparsers.add_parser("notations", help="Show notation registry (B.3)")
    p_notations.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_notations.add_argument("--limit", "-n", type=int, default=20, help="Max notations to show")
    p_notations.set_defaults(func=cmd_notations)

    # breakthroughs
    p_breakthroughs = subparsers.add_parser("breakthroughs", help="Show breakthrough sessions (B.3)")
    p_breakthroughs.add_argument("--thread", "-t", default="eternal", help="Thread ID")
    p_breakthroughs.set_defaults(func=cmd_breakthroughs)

    # === Papers subcommands (B.1) ===
    p_papers = subparsers.add_parser("papers", help="Paper corpus management")
    papers_sub = p_papers.add_subparsers(dest="papers_command", help="Papers command")

    # papers register
    p_papers_reg = papers_sub.add_parser("register", help="Register a paper")
    p_papers_reg.add_argument("--arxiv", required=True, help="Arxiv paper ID")
    p_papers_reg.add_argument("--name", required=True, help="Short name (e.g., Vec2Text)")
    p_papers_reg.add_argument("--title", help="Paper title")
    p_papers_reg.add_argument("--category", required=True, help="Category (vec2text, hdc_vsa, etc.)")
    p_papers_reg.add_argument("--pdf", help="Path to PDF file")
    p_papers_reg.set_defaults(func=cmd_papers_register)

    # papers convert
    p_papers_conv = papers_sub.add_parser("convert", help="Set markdown path after conversion")
    p_papers_conv.add_argument("--arxiv", required=True, help="Arxiv paper ID")
    p_papers_conv.add_argument("--markdown", required=True, help="Path to markdown file")
    p_papers_conv.set_defaults(func=cmd_papers_convert)

    # papers index
    p_papers_idx = papers_sub.add_parser("index", help="Index a paper")
    p_papers_idx.add_argument("--arxiv", help="Arxiv paper ID")
    p_papers_idx.add_argument("--all", action="store_true", help="Index all converted papers")
    p_papers_idx.set_defaults(func=cmd_papers_index)

    # papers list
    p_papers_list = papers_sub.add_parser("list", help="List papers")
    p_papers_list.add_argument("--category", help="Filter by category")
    p_papers_list.add_argument("--status", help="Filter by status")
    p_papers_list.set_defaults(func=cmd_papers_list)

    # papers query
    p_papers_query = papers_sub.add_parser("query", help="Query papers by E (Born rule)")
    p_papers_query.add_argument("query", help="Query text")
    p_papers_query.add_argument("--k", type=int, default=10, help="Number of results")
    p_papers_query.add_argument("--preview", action="store_true", help="Show content preview")
    p_papers_query.set_defaults(func=cmd_papers_query)

    # papers status
    p_papers_stat = papers_sub.add_parser("status", help="Show corpus status")
    p_papers_stat.set_defaults(func=cmd_papers_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Handle papers subcommand
    if args.command == "papers":
        if not hasattr(args, 'func') or args.func is None:
            p_papers.print_help()
            return

    try:
        args.func(args)
    finally:
        global _resident
        if _resident is not None:
            _resident.close()


if __name__ == "__main__":
    main()
