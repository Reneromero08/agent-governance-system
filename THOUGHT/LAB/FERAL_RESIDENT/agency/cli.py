#!/usr/bin/env python3
"""
Feral Resident CLI (A.4.1 + B.3 + P.1 + P.2 + P.3)

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

Swarm Integration (P.1):
    feral swarm start --residents alpha:dolphin3 beta:ministral-3b
    feral swarm status
    feral swarm switch alpha
    feral swarm think "What is authentication?"
    feral swarm broadcast "What is authentication?"
    feral swarm observe
    feral swarm stop

Symbolic Compiler (P.2):
    feral compile render "concept text" --level 2
    feral compile all "concept text"
    feral compile decompress "[v:hash]" --level 2
    feral compile verify <hash> <compressed> --level 2
    feral compile stats

Catalytic Closure (P.3):
    feral closure status
    feral closure prove --thought-hash abc123 --resident alpha
    feral closure verify-chain --from receipt1 --to receipt2
    feral closure check-df --resident alpha
    feral closure patterns --min-freq 3
    feral closure efficiency --window 100
    feral closure cache-stats
    feral closure register-form --name "MyForm" --text "concept"
    feral closure optimize
    feral closure gates
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add imports path
FERAL_PATH = Path(__file__).parent.parent  # agency/ -> FERAL_RESIDENT/
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from cognition.vector_brain import VectorResident, ResidentBenchmark


# Global resident instance for session
# Lazy import for symbolic compiler to avoid loading unless needed
_symbolic_compiler = None


def get_symbolic_compiler():
    """Get or create symbolic compiler instance"""
    global _symbolic_compiler
    if _symbolic_compiler is None:
        from emergence.symbolic_compiler import create_compiler
        # Create with basic corpus
        corpus = ["test concept", "semantic meaning", "vector representation"]
        _symbolic_compiler = create_compiler(corpus=corpus)
    return _symbolic_compiler


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


# === Swarm Integration (P.1) ===

# Global swarm coordinator instance
_swarm_coordinator = None


def get_swarm_coordinator():
    """Get or create swarm coordinator"""
    global _swarm_coordinator
    if _swarm_coordinator is None:
        from swarm_coordinator import SwarmCoordinator
        _swarm_coordinator = SwarmCoordinator.load_active()
    return _swarm_coordinator


def cmd_swarm_start(args):
    """Start multiple residents in swarm mode"""
    from swarm_coordinator import SwarmCoordinator

    global _swarm_coordinator

    # Parse resident specs: "name:model" format
    configs = []
    for spec in args.residents:
        parts = spec.split(':')
        name = parts[0]
        model = parts[1] if len(parts) > 1 else "dolphin3:latest"
        configs.append({
            'name': name,
            'model': model
        })

    _swarm_coordinator = SwarmCoordinator()
    status = _swarm_coordinator.start_swarm(configs)

    print(f"=== Swarm Started ===")
    print(f"Residents: {status.resident_count}")
    for name, r_status in status.residents.items():
        active = " [ACTIVE]" if r_status.is_active else ""
        print(f"  [{name}]{active} model={r_status.model}, thread={r_status.thread_id}")
    print(f"Shared space: {status.shared_space.get('db_path', 'unknown')}")


def cmd_swarm_stop(args):
    """Stop the swarm"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm.")
        return

    coordinator.stop_swarm()
    print("Swarm stopped.")


def cmd_swarm_status(args):
    """Show swarm status"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm. Start with 'feral swarm start'")
        return

    coordinator.print_status()


def cmd_swarm_switch(args):
    """Switch active resident for interaction"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm. Start with 'feral swarm start'")
        return

    try:
        coordinator.set_active_resident(args.resident)
        print(f"Active resident: {args.resident}")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available: {list(coordinator.residents.keys())}")


def cmd_swarm_think(args):
    """Send query to active resident"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm. Start with 'feral swarm start'")
        return

    try:
        result = coordinator.think(args.query)
        print(f"Q: {args.query}")
        print(f"A: {result.response}")
        print()
        print(f"  E_resonance: {result.E_resonance:.3f}")
        print(f"  E_compression: {result.E_compression:.3f}")
        print(f"  Mind Df: {result.mind_Df:.1f}")
        print(f"  Distance: {result.distance_from_start:.3f}")
    except RuntimeError as e:
        print(f"Error: {e}")


def cmd_swarm_broadcast(args):
    """Send same query to all residents"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm. Start with 'feral swarm start'")
        return

    broadcast_result = coordinator.broadcast_think(args.query)

    print(f"=== Broadcast: '{args.query}' ===")
    print()

    for name, result in broadcast_result.results.items():
        print(f"[{name}]")
        response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
        print(f"  Response: {response_preview}")
        print(f"  E: {result.E_resonance:.3f}, Df: {result.mind_Df:.1f}")
        print()

    if broadcast_result.convergence_observed:
        print("(Convergence observation recorded)")


def cmd_swarm_observe(args):
    """Observe convergence between residents"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm. Start with 'feral swarm start'")
        return

    if len(coordinator.residents) < 2:
        print("Need at least 2 residents to observe convergence.")
        return

    summary = coordinator.observe_convergence()

    print(f"=== Convergence Observation ===")
    print(f"Timestamp: {summary.timestamp}")
    print(f"Residents: {summary.resident_count}")
    print(f"Pairs: {summary.pair_count}")
    print()

    for pair_id, metrics in summary.pairs.items():
        print(f"Pair: {pair_id}")
        print(f"  E(mind_A, mind_B): {metrics.E_mind_correlation:.4f}")
        print(f"  Df correlation:   {metrics.Df_correlation:.4f}")
        print(f"  Shared notations: {metrics.shared_notation_count}")
        print(f"  Df_A: {metrics.Df_a:.1f}, Df_B: {metrics.Df_b:.1f}")
        print()

    print(f"Summary:")
    print(f"  Mean E: {summary.E_minds_mean:.4f}")
    print(f"  Max E:  {summary.E_minds_max:.4f}")
    print(f"  Min E:  {summary.E_minds_min:.4f}")
    print(f"  Total convergence events: {summary.total_convergence_events}")
    print(f"  Total shared notations: {summary.total_shared_notations}")


def cmd_swarm_history(args):
    """Show convergence history"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm.")
        return

    events = coordinator.get_convergence_history(limit=args.limit)

    print(f"=== Convergence History (last {len(events)}) ===")
    print()

    if not events:
        print("No convergence events recorded yet.")
        return

    for event in events:
        print(f"[{event['event_type']}] {event['timestamp'][:19]}")
        print(f"  {event['resident_a']} <-> {event['resident_b']}")
        print(f"  E: {event['E_value']:.4f}")
        print(f"  Df: {event['Df_a']:.1f} / {event['Df_b']:.1f}")
        print()


def cmd_swarm_add(args):
    """Add a resident to running swarm"""
    coordinator = get_swarm_coordinator()
    if coordinator is None:
        print("No active swarm. Start with 'feral swarm start'")
        return

    parts = args.spec.split(':')
    name = parts[0]
    model = parts[1] if len(parts) > 1 else "dolphin3:latest"

    try:
        status = coordinator.add_resident({'name': name, 'model': model})
        print(f"Added resident: {status.name}")
        print(f"  Model: {status.model}")
        print(f"  Thread: {status.thread_id}")
    except ValueError as e:
        print(f"Error: {e}")


# =============================================================================
# Symbolic Compiler Commands (P.2)
# =============================================================================

def cmd_compile(args):
    """Compile text to specified compression level"""
    compiler = get_symbolic_compiler()

    # Initialize text to GeometricState
    state = compiler.reasoner.initialize(args.text)

    # Render at specified level
    result = compiler.render(state, args.level, resident_id=args.resident)

    print(f"=== Compile to Level {args.level} ({result.level_name}) ===")
    print(f"Input: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print(f"Output: {result.content}")
    print()
    print(f"  Compression ratio: {result.compression_ratio:.2f}x")
    print(f"  Original hash: {result.original_hash}")
    print(f"  Receipt: {result.receipt_hash}")

    if args.verify:
        verification = compiler.verify_roundtrip(state, result.content, args.level, args.resident)
        print()
        print(f"Round-trip verification:")
        print(f"  E delta: {verification.E_delta:.6f}")
        print(f"  Df delta: {verification.Df_delta:.6f}")
        print(f"  Verified: {'PASS' if verification.verified else 'FAIL'}")


def cmd_decompress(args):
    """Decompress compressed form back to prose"""
    compiler = get_symbolic_compiler()

    # Decompress
    state = compiler.decompress(args.content, args.level, resident_id=args.resident)

    if state is None:
        print(f"Error: Could not decompress content at level {args.level}")
        print("Note: The original state must be cached from a previous compile operation.")
        return

    # Render as prose
    result = compiler.render(state, 0)  # Level 0 = prose

    print(f"=== Decompress from Level {args.level} ===")
    print(f"Input: {args.content}")
    print(f"Output: {result.content}")
    print()
    print(f"  State Df: {state.Df:.2f}")
    print(f"  Hash: {result.original_hash}")


def cmd_verify(args):
    """Verify lossless round-trip for compressed content"""
    compiler = get_symbolic_compiler()

    # Need to get original state from hash
    # First check if we have the state cached
    if args.original_hash in compiler._state_cache:
        original = compiler._state_cache[args.original_hash]
    else:
        print(f"Error: Original state {args.original_hash} not found in cache.")
        print("Note: Run 'compile' first to cache the state.")
        return

    verification = compiler.verify_roundtrip(
        original, args.compressed, args.level, resident_id=args.resident
    )

    print(f"=== Round-Trip Verification ===")
    print(f"Original hash: {verification.original_hash}")
    print(f"Compressed: {args.compressed}")
    print(f"Level: {args.level}")
    print()
    print(f"  Decompressed hash: {verification.decompressed_hash}")
    print(f"  E delta: {verification.E_delta:.6f} (threshold: < 0.01)")
    print(f"  Df delta: {verification.Df_delta:.6f} (threshold: < 0.01)")
    print()
    if verification.verified:
        print("  VERIFIED: Lossless round-trip confirmed (E > 0.99)")
    else:
        print("  FAILED: Round-trip not lossless")
    print()
    print(f"  Receipt: {verification.receipt.get('receipt_hash', 'N/A')}")


def cmd_compression_stats(args):
    """Show compression statistics"""
    compiler = get_symbolic_compiler()
    stats = compiler.get_compression_stats()

    print(f"=== P.2 Symbolic Compiler Statistics ===")
    print()
    print("Renders by level:")
    for level in range(4):
        count = stats['renders_by_level'].get(level, 0)
        level_name = ['prose', 'symbol', 'hash', 'protocol'][level]
        print(f"  Level {level} ({level_name}): {count}")
    print(f"  Total: {stats['total_renders']}")
    print()
    print("Verifications:")
    print(f"  Total: {stats['total_verifications']}")
    print(f"  Passed: {stats['verified_count']}")
    print(f"  Failed: {stats['failed_count']}")
    print(f"  Rate: {stats['verification_rate']:.1%}")
    print()
    print("Symbol Registry:")
    print(f"  Global symbols: {stats['registry_global_count']}")
    print(f"  Local symbols: {stats['registry_local_count']}")

    if args.json:
        import json
        print()
        print("JSON:")
        print(json.dumps(stats, indent=2))


def cmd_compile_levels(args):
    """Show all compression levels for text"""
    compiler = get_symbolic_compiler()

    # Initialize text to GeometricState
    state = compiler.reasoner.initialize(args.text)

    print(f"=== Multi-Level Rendering ===")
    print(f"Input: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"State Df: {state.Df:.2f}")
    print()

    for level in range(4):
        result = compiler.render(state, level, resident_id=args.resident)
        print(f"Level {level} ({result.level_name}):")
        print(f"  {result.content}")
        print(f"  Compression: {result.compression_ratio:.2f}x | Receipt: {result.receipt_hash}")
        print()


# =============================================================================
# Catalytic Closure Commands (P.3)
# =============================================================================

# Global closure instance (lazy-loaded)
_catalytic_closure = None


def get_catalytic_closure():
    """Get or create catalytic closure instance"""
    global _catalytic_closure
    if _catalytic_closure is None:
        from catalytic_closure import CatalyticClosure
        _catalytic_closure = CatalyticClosure()
    return _catalytic_closure


def cmd_closure_status(args):
    """Show P.3 Catalytic Closure status"""
    closure = get_catalytic_closure()
    status = closure.get_status()

    print("=== P.3 Catalytic Closure Status ===")
    print()
    print("Merkle Chain:")
    print(f"  Length: {status['merkle_chain_length']}")
    print(f"  Root: {status['merkle_root'] or '(empty)'}")
    print()
    print("Self-Optimization:")
    print(f"  Cached compositions: {status['cached_compositions']}")
    print()
    print("Meta-Operations:")
    print(f"  Canonical forms: {status['canonical_forms']}")
    print(f"  Custom gates: {status['custom_gates']}")
    print()
    print("Navigation Parameters:")
    for param, value in status['navigation_params'].items():
        print(f"  {param}: {value}")


def cmd_closure_prove(args):
    """Prove thought authenticity (P.3.3)"""
    closure = get_catalytic_closure()

    proof = closure.prove_thought(args.thought_hash, args.resident)

    print("=== Thought Authenticity Proof ===")
    print()
    print(f"Thought hash: {proof.thought_hash}")
    print(f"Resident: {proof.resident_id}")
    print()

    if proof.is_authentic:
        print("AUTHENTIC")
        print()
        print(f"  E at creation: {proof.E_at_creation:.4f}")
        print(f"  Df at creation: {proof.Df_at_creation:.2f}")
        print(f"  Continuity verified: {proof.continuity_verified}")
        print(f"  Merkle proof length: {len(proof.merkle_proof)}")
        print(f"  Receipt chain length: {len(proof.receipt_chain)}")
        print()
        print(f"Proof hash: {proof.proof_hash}")
    else:
        print("NOT AUTHENTIC or NOT FOUND")
        if proof.receipt_chain:
            print(f"  Reason: {proof.receipt_chain[0].get('error', 'unknown')}")


def cmd_closure_verify_chain(args):
    """Verify receipt chain integrity (P.3.3)"""
    closure = get_catalytic_closure()

    verification = closure.verify_chain(args.start, args.end)

    print("=== Receipt Chain Verification ===")
    print()
    print(f"Start: {verification.start_hash}")
    print(f"End: {verification.end_hash}")
    print(f"Chain length: {verification.chain_length}")
    print()

    if verification.is_valid:
        print("VALID - Chain is unbroken")
    else:
        print("INVALID - Chain has issues")
        if verification.gaps:
            print("Gaps found:")
            for gap in verification.gaps:
                print(f"  {gap}")
        if verification.tampering_detected:
            print("TAMPERING DETECTED")

    print()
    print(f"Verification receipt: {verification.verification_receipt}")


def cmd_closure_check_df(args):
    """Check Df continuity for resident (P.3.3)"""
    closure = get_catalytic_closure()

    report = closure.check_df_continuity(args.resident)

    print("=== Df Continuity Report ===")
    print()
    print(f"Resident: {report.resident_id}")
    print(f"Samples: {report.sample_count}")
    print()

    if report.is_continuous:
        print("CONTINUOUS - No anomalies detected")
    else:
        print("ANOMALIES DETECTED")

    print()
    print(f"Max delta observed: {report.max_delta_observed:.4f}")
    print(f"Average delta: {report.average_delta:.4f}")

    if report.anomalies:
        print()
        print("Anomalies:")
        for anomaly in report.anomalies[:5]:
            print(f"  [{anomaly['type']}] index={anomaly.get('index', '?')}, "
                  f"from={anomaly.get('from_Df', '?'):.2f}, to={anomaly.get('to_Df', '?'):.2f}")


def cmd_closure_patterns(args):
    """Detect optimization patterns (P.3.2)"""
    closure = get_catalytic_closure()

    patterns = closure.detect_patterns(min_frequency=args.min_freq)

    print("=== Optimization Patterns ===")
    print()

    if not patterns:
        print("No patterns detected yet.")
        print("Run more operations to build pattern history.")
        return

    print(f"Patterns found: {len(patterns)}")
    print()

    for i, pattern in enumerate(patterns[:args.limit]):
        print(f"[{i+1}] {pattern.pattern_type}: {pattern.description}")
        print(f"    Frequency: {pattern.frequency}")
        print(f"    Potential savings: {pattern.potential_savings} ops")
        print(f"    Suggested: {pattern.suggested_action}")
        print()


def cmd_closure_efficiency(args):
    """Show efficiency report (P.3.2)"""
    closure = get_catalytic_closure()

    report = closure.get_efficiency_report(window=args.window)

    print(f"=== Efficiency Report (window={report.window_size}) ===")
    print()
    print(f"Ops per interaction: {report.ops_per_interaction:.2f}")
    print(f"Cache hit rate: {report.cache_hit_rate:.1%}")
    print(f"Navigation depth avg: {report.navigation_depth_avg:.2f}")
    print(f"E stability: {report.E_stability:.3f}")
    print()
    print(f"Improvement trend: {report.improvement_trend:+.1%}")
    if report.improvement_trend > 0:
        print("  (System is getting more efficient)")
    elif report.improvement_trend < 0:
        print("  (System efficiency declining)")
    else:
        print("  (No change)")
    print()
    print(f"Receipt: {report.receipt_hash}")


def cmd_closure_cache_stats(args):
    """Show composition cache statistics (P.3.2)"""
    closure = get_catalytic_closure()

    stats = closure.get_cache_stats()

    print("=== Composition Cache Statistics ===")
    print()
    print(f"Cached compositions: {stats['cached_compositions']}")
    print(f"Tracked compositions: {stats['tracked_compositions']}")
    print(f"Total observations: {stats['total_observations']}")

    if args.json:
        print()
        print("JSON:")
        print(json.dumps(stats, indent=2))


def cmd_closure_register_form(args):
    """Register a canonical form (P.3.1)"""
    closure = get_catalytic_closure()

    # Initialize text to GeometricState
    state = closure.canonical_registry.reasoner.initialize(args.text)

    receipt = closure.register_canonical(
        state=state,
        name=args.name,
        justification=args.justification or f"Registered via CLI: {args.text[:50]}"
    )

    print("=== Canonical Form Registration ===")
    print()
    print(f"Name: {receipt.name}")
    print(f"State hash: {receipt.state_hash}")
    print()

    if "REJECTED" in receipt.justification:
        print(f"REJECTED: {receipt.justification}")
    else:
        print("REGISTERED")
        print(f"  Coherence E: {receipt.coherence_E:.4f}")
        print(f"  Session count: {receipt.session_count}/{closure.canonical_registry.MAX_FORMS_PER_SESSION}")
        print(f"  Merkle proof: {receipt.merkle_proof}")
        print(f"  Receipt: {receipt.receipt_hash}")


def cmd_closure_optimize(args):
    """Suggest navigation optimizations (P.3.1)"""
    closure = get_catalytic_closure()

    suggestions = closure.suggest_optimizations()

    print("=== Navigation Optimization Suggestions ===")
    print()

    if not suggestions:
        print("No optimizations suggested at this time.")
        print("Run more interactions to gather optimization data.")
        return

    print(f"Suggestions: {len(suggestions)}")
    print()

    for i, suggestion in enumerate(suggestions):
        print(f"[{i+1}] Parameter: {suggestion.parameter}")
        print(f"    Current: {suggestion.current_value}")
        print(f"    Suggested: {suggestion.suggested_value}")
        print(f"    Expected improvement: {suggestion.expected_improvement:.1%}")
        print(f"    Confidence: {suggestion.confidence:.1%}")
        print()

    if args.apply:
        print("Applying first suggestion...")
        if suggestions:
            success = closure.navigation_optimizer.apply_optimization(suggestions[0])
            if success:
                print(f"Applied: {suggestions[0].parameter} = {suggestions[0].suggested_value}")
            else:
                print("Failed to apply optimization")


def cmd_closure_gates(args):
    """List custom gates (P.3.1)"""
    closure = get_catalytic_closure()

    gates = closure.gate_definer.list_gates()

    print("=== Custom Gates ===")
    print()

    if not gates:
        print("No custom gates defined yet.")
        print("Use the Python API to define custom gates.")
        return

    print(f"Total gates: {len(gates)}")
    print()

    for gate in gates:
        print(f"Gate: {gate['name']}")
        print(f"  Description: {gate.get('description', 'N/A')}")
        print(f"  Test passed: {gate.get('test_passed', False)}")
        print(f"  E preservation: {gate.get('E_preservation', 0):.4f}")
        print(f"  Defined: {gate.get('defined_at', 'unknown')}")
        print()


def cmd_closure_forms(args):
    """List canonical forms (P.3.1)"""
    closure = get_catalytic_closure()

    forms = closure.canonical_registry.list_forms()

    print("=== Canonical Forms ===")
    print()

    if not forms:
        print("No canonical forms registered yet.")
        return

    print(f"Total forms: {len(forms)}")
    print()

    for form in forms[:args.limit]:
        print(f"Form: {form['name']}")
        print(f"  Hash: {form['state_hash']}")
        print(f"  Coherence E: {form['coherence_E']:.4f}")
        print(f"  Df: {form['Df']:.2f}")
        print(f"  Registered: {form['registered_at']}")
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

Swarm integration (P.1):
  python cli.py swarm start --residents alpha:dolphin3 beta:ministral-3b
  python cli.py swarm status
  python cli.py swarm switch alpha
  python cli.py swarm think "What is authentication?"
  python cli.py swarm broadcast "What is authentication?"
  python cli.py swarm observe
  python cli.py swarm stop
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

    # === Swarm subcommands (P.1) ===
    p_swarm = subparsers.add_parser("swarm", help="Swarm operations (P.1)")
    swarm_sub = p_swarm.add_subparsers(dest="swarm_command", help="Swarm command")

    # swarm start
    p_swarm_start = swarm_sub.add_parser("start", help="Start multi-resident swarm")
    p_swarm_start.add_argument(
        "--residents", "-r", nargs="+", required=True,
        help="Resident specs: 'name:model' (e.g., 'alpha:dolphin3 beta:ministral-3b')"
    )
    p_swarm_start.set_defaults(func=cmd_swarm_start)

    # swarm stop
    p_swarm_stop = swarm_sub.add_parser("stop", help="Stop the swarm")
    p_swarm_stop.set_defaults(func=cmd_swarm_stop)

    # swarm status
    p_swarm_status = swarm_sub.add_parser("status", help="Show swarm status")
    p_swarm_status.set_defaults(func=cmd_swarm_status)

    # swarm switch
    p_swarm_switch = swarm_sub.add_parser("switch", help="Switch active resident")
    p_swarm_switch.add_argument("resident", help="Resident name to activate")
    p_swarm_switch.set_defaults(func=cmd_swarm_switch)

    # swarm think
    p_swarm_think = swarm_sub.add_parser("think", help="Send query to active resident")
    p_swarm_think.add_argument("query", help="Query to think about")
    p_swarm_think.set_defaults(func=cmd_swarm_think)

    # swarm broadcast
    p_swarm_broadcast = swarm_sub.add_parser("broadcast", help="Send query to all residents")
    p_swarm_broadcast.add_argument("query", help="Query to broadcast")
    p_swarm_broadcast.set_defaults(func=cmd_swarm_broadcast)

    # swarm observe
    p_swarm_observe = swarm_sub.add_parser("observe", help="Observe convergence between residents")
    p_swarm_observe.set_defaults(func=cmd_swarm_observe)

    # swarm history
    p_swarm_history = swarm_sub.add_parser("history", help="Show convergence history")
    p_swarm_history.add_argument("--limit", "-n", type=int, default=20, help="Max events to show")
    p_swarm_history.set_defaults(func=cmd_swarm_history)

    # swarm add
    p_swarm_add = swarm_sub.add_parser("add", help="Add resident to running swarm")
    p_swarm_add.add_argument("spec", help="Resident spec: 'name:model'")
    p_swarm_add.set_defaults(func=cmd_swarm_add)

    # === Symbolic Compiler subcommands (P.2) ===
    p_compile = subparsers.add_parser("compile", help="Symbolic compiler operations (P.2)")
    compile_sub = p_compile.add_subparsers(dest="compile_command", help="Compile command")

    # compile render
    p_compile_render = compile_sub.add_parser("render", help="Render text at compression level")
    p_compile_render.add_argument("text", help="Text to compile")
    p_compile_render.add_argument("--level", "-l", type=int, default=2, choices=[0, 1, 2, 3],
                                  help="Compression level (0=prose, 1=symbol, 2=hash, 3=protocol)")
    p_compile_render.add_argument("--resident", "-r", default="default", help="Resident ID")
    p_compile_render.add_argument("--verify", "-v", action="store_true", help="Verify round-trip")
    p_compile_render.set_defaults(func=cmd_compile)

    # compile all
    p_compile_all = compile_sub.add_parser("all", help="Show all compression levels for text")
    p_compile_all.add_argument("text", help="Text to compile")
    p_compile_all.add_argument("--resident", "-r", default="default", help="Resident ID")
    p_compile_all.set_defaults(func=cmd_compile_levels)

    # compile decompress
    p_compile_decomp = compile_sub.add_parser("decompress", help="Decompress to prose")
    p_compile_decomp.add_argument("content", help="Compressed content")
    p_compile_decomp.add_argument("--level", "-l", type=int, required=True, choices=[0, 1, 2, 3],
                                  help="Source compression level")
    p_compile_decomp.add_argument("--resident", "-r", default="default", help="Resident ID")
    p_compile_decomp.set_defaults(func=cmd_decompress)

    # compile verify
    p_compile_verify = compile_sub.add_parser("verify", help="Verify lossless round-trip")
    p_compile_verify.add_argument("original_hash", help="Original state hash")
    p_compile_verify.add_argument("compressed", help="Compressed content")
    p_compile_verify.add_argument("--level", "-l", type=int, required=True, choices=[0, 1, 2, 3],
                                  help="Compression level")
    p_compile_verify.add_argument("--resident", "-r", default="default", help="Resident ID")
    p_compile_verify.set_defaults(func=cmd_verify)

    # compile stats
    p_compile_stats = compile_sub.add_parser("stats", help="Show compression statistics")
    p_compile_stats.add_argument("--json", action="store_true", help="Output as JSON")
    p_compile_stats.set_defaults(func=cmd_compression_stats)

    # === Catalytic Closure subcommands (P.3) ===
    p_closure = subparsers.add_parser("closure", help="Catalytic closure operations (P.3)")
    closure_sub = p_closure.add_subparsers(dest="closure_command", help="Closure command")

    # closure status
    p_closure_status = closure_sub.add_parser("status", help="Show P.3 status")
    p_closure_status.set_defaults(func=cmd_closure_status)

    # closure prove
    p_closure_prove = closure_sub.add_parser("prove", help="Prove thought authenticity (P.3.3)")
    p_closure_prove.add_argument("--thought-hash", required=True, help="Hash of thought to prove")
    p_closure_prove.add_argument("--resident", "-r", required=True, help="Resident ID")
    p_closure_prove.set_defaults(func=cmd_closure_prove)

    # closure verify-chain
    p_closure_verify = closure_sub.add_parser("verify-chain", help="Verify receipt chain (P.3.3)")
    p_closure_verify.add_argument("--start", required=True, help="Start receipt hash")
    p_closure_verify.add_argument("--end", required=True, help="End receipt hash")
    p_closure_verify.set_defaults(func=cmd_closure_verify_chain)

    # closure check-df
    p_closure_df = closure_sub.add_parser("check-df", help="Check Df continuity (P.3.3)")
    p_closure_df.add_argument("--resident", "-r", required=True, help="Resident ID")
    p_closure_df.set_defaults(func=cmd_closure_check_df)

    # closure patterns
    p_closure_patterns = closure_sub.add_parser("patterns", help="Detect optimization patterns (P.3.2)")
    p_closure_patterns.add_argument("--min-freq", type=int, default=3, help="Minimum frequency threshold")
    p_closure_patterns.add_argument("--limit", "-n", type=int, default=10, help="Max patterns to show")
    p_closure_patterns.set_defaults(func=cmd_closure_patterns)

    # closure efficiency
    p_closure_eff = closure_sub.add_parser("efficiency", help="Show efficiency report (P.3.2)")
    p_closure_eff.add_argument("--window", "-w", type=int, default=100, help="Window size for metrics")
    p_closure_eff.set_defaults(func=cmd_closure_efficiency)

    # closure cache-stats
    p_closure_cache = closure_sub.add_parser("cache-stats", help="Show cache statistics (P.3.2)")
    p_closure_cache.add_argument("--json", action="store_true", help="Output as JSON")
    p_closure_cache.set_defaults(func=cmd_closure_cache_stats)

    # closure register-form
    p_closure_register = closure_sub.add_parser("register-form", help="Register canonical form (P.3.1)")
    p_closure_register.add_argument("--name", required=True, help="Name for the form")
    p_closure_register.add_argument("--text", required=True, help="Text to initialize as form")
    p_closure_register.add_argument("--justification", "-j", help="Justification for registration")
    p_closure_register.set_defaults(func=cmd_closure_register_form)

    # closure optimize
    p_closure_opt = closure_sub.add_parser("optimize", help="Suggest navigation optimizations (P.3.1)")
    p_closure_opt.add_argument("--apply", action="store_true", help="Apply first suggestion")
    p_closure_opt.set_defaults(func=cmd_closure_optimize)

    # closure gates
    p_closure_gates = closure_sub.add_parser("gates", help="List custom gates (P.3.1)")
    p_closure_gates.set_defaults(func=cmd_closure_gates)

    # closure forms
    p_closure_forms = closure_sub.add_parser("forms", help="List canonical forms (P.3.1)")
    p_closure_forms.add_argument("--limit", "-n", type=int, default=20, help="Max forms to show")
    p_closure_forms.set_defaults(func=cmd_closure_forms)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Handle papers subcommand
    if args.command == "papers":
        if not hasattr(args, 'func') or args.func is None:
            p_papers.print_help()
            return

    # Handle swarm subcommand
    if args.command == "swarm":
        if not hasattr(args, 'func') or args.func is None:
            p_swarm.print_help()
            return

    # Handle compile subcommand (P.2)
    if args.command == "compile":
        if not hasattr(args, 'func') or args.func is None:
            p_compile.print_help()
            return

    # Handle closure subcommand (P.3)
    if args.command == "closure":
        if not hasattr(args, 'func') or args.func is None:
            p_closure.print_help()
            return

    try:
        args.func(args)
    finally:
        global _resident
        global _swarm_coordinator
        if _resident is not None:
            _resident.close()
        if _swarm_coordinator is not None:
            _swarm_coordinator.close()


if __name__ == "__main__":
    main()
