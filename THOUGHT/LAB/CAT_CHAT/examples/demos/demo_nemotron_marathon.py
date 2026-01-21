#!/usr/bin/env python3
"""
CATALYTIC CONTEXT DEMO: Nemotron Marathon
==========================================

This demo showcases the paradigm-shifting catalytic context management
with a real 30B parameter LLM (nemotron-3-nano-30b-a3b).

Features demonstrated:
1. Automatic context compression every turn
2. Semantic rehydration when old content becomes relevant
3. Budget management under real LLM responses
4. Live metrics and decay analysis

Requirements:
- LLM Studio running at http://10.5.0.2:1234 with:
  - nemotron-3-nano-30b-a3b
  - text-embedding-nomic-embed-text-v1.5

Usage:
    python demo_nemotron_marathon.py [--turns 50] [--interactive]

Author: Catalytic Computing Research
"""

import argparse
import sys
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Add parent to path for imports
CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import (
    SessionCapsule,
    EVENT_PARTITION,
    EVENT_TURN_STORED,
    EVENT_TURN_HYDRATED,
)
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.context_partitioner import ContextItem


# =============================================================================
# Configuration
# =============================================================================

LLM_STUDIO_BASE = "http://10.5.0.2:1234"
NEMOTRON_MODEL = "nemotron-3-nano-30b-a3b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CONTEXT_WINDOW = 40961


# =============================================================================
# API Functions
# =============================================================================

def check_llm_studio() -> Tuple[bool, str]:
    """Check if LLM Studio is available."""
    try:
        resp = requests.get(f"{LLM_STUDIO_BASE}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            if NEMOTRON_MODEL in models and EMBEDDING_MODEL in models:
                return True, f"Connected: {NEMOTRON_MODEL}, {EMBEDDING_MODEL}"
            return False, f"Missing models. Available: {models}"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from nomic model."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    embedding = np.array(resp.json()["data"][0]["embedding"])
    return embedding / np.linalg.norm(embedding)


def generate_response(system: str, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from nemotron."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/chat/completions",
        json={
            "model": NEMOTRON_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# =============================================================================
# Demo Scenarios
# =============================================================================

@dataclass
class DemoScenario:
    """A demo conversation scenario."""
    name: str
    facts: List[Tuple[int, str]]  # (turn, fact)
    filler_topic: str
    recall_queries: List[str]


SCENARIOS = {
    "physics": DemoScenario(
        name="Physics Knowledge Chain",
        facts=[
            (5, "The gravitational constant G equals 6.674e-11 N*m^2/kg^2."),
            (15, "The speed of light c equals 299,792,458 m/s exactly."),
            (25, "The Schwarzschild radius formula is r_s = 2GM/c^2."),
        ],
        filler_topic="cooking and recipes",
        recall_queries=[
            "What is the gravitational constant?",
            "What is the speed of light?",
            "What is the Schwarzschild radius formula?",
        ],
    ),
    "history": DemoScenario(
        name="Historical Facts Chain",
        facts=[
            (5, "The Declaration of Independence was signed on July 4, 1776."),
            (15, "World War II ended on September 2, 1945."),
            (25, "The Moon landing occurred on July 20, 1969."),
        ],
        filler_topic="gardening and plants",
        recall_queries=[
            "When was the Declaration of Independence signed?",
            "When did World War II end?",
            "When did the Moon landing occur?",
        ],
    ),
    "math": DemoScenario(
        name="Mathematical Constants",
        facts=[
            (5, "Euler's number e equals approximately 2.71828."),
            (15, "Pi equals approximately 3.14159."),
            (25, "The golden ratio phi equals approximately 1.61803."),
        ],
        filler_topic="sports and athletics",
        recall_queries=[
            "What is Euler's number?",
            "What is the value of pi?",
            "What is the golden ratio?",
        ],
    ),
}


# =============================================================================
# Demo Runner
# =============================================================================

def run_demo(
    scenario: DemoScenario,
    total_turns: int = 50,
    interactive: bool = False,
    verbose: bool = True
):
    """Run the catalytic context demo."""
    import tempfile

    print("\n" + "=" * 70)
    print(f"CATALYTIC CONTEXT DEMO: {scenario.name}")
    print("=" * 70)
    print(f"Model: {NEMOTRON_MODEL} ({CONTEXT_WINDOW} tokens)")
    print(f"Scenario: Plant {len(scenario.facts)} facts, bury in filler, then recall")
    print("=" * 70 + "\n")

    # Setup
    tmpdir = Path(tempfile.gettempdir()) / "catalytic_demo_marathon"
    tmpdir.mkdir(exist_ok=True)
    db_path = tmpdir / f"demo_{scenario.name.lower().replace(' ', '_')}.db"

    if db_path.exists():
        db_path.unlink()

    capsule = SessionCapsule(db_path=db_path)
    session_id = capsule.create_session()

    system_prompt = "You are a knowledgeable assistant. Remember facts precisely."
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=CONTEXT_WINDOW,
        system_prompt=system_prompt,
        response_reserve_pct=0.25,
        model_id=NEMOTRON_MODEL,
    )

    manager = AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=get_embedding,
        E_threshold=0.3,
    )
    manager.capsule = capsule

    print(f"Budget: {budget.available_for_working_set} tokens for working set")
    print(f"Session: {session_id}")
    print("\n" + "-" * 70)

    # Prepare fact schedule
    fact_turns = {t: f for t, f in scenario.facts}
    max_fact_turn = max(t for t, _ in scenario.facts)

    # Phase 1: Run turns
    print("\nPHASE 1: Building conversation with planted facts...\n")
    start_time = time.time()

    for turn in range(1, total_turns + 1):
        if turn in fact_turns:
            # Plant a fact
            query = f"Remember this important fact: {fact_turns[turn]}"
            print(f"Turn {turn}: [FACT] {fact_turns[turn][:60]}...")
        else:
            # Filler
            query = f"Tell me something interesting about {scenario.filler_topic}."
            if verbose and turn % 10 == 0:
                print(f"Turn {turn}: [filler about {scenario.filler_topic}]")

        result = manager.respond_catalytic(
            query=query,
            llm_generate=lambda s, p: generate_response(s, p),
            system_prompt=system_prompt,
        )

        if interactive and turn in fact_turns:
            print(f"  Response: {result.response[:100]}...")
            print(f"  E_mean: {result.E_mean:.3f}, Compression: {result.compression_ratio:.1f}x")

    elapsed = time.time() - start_time
    print(f"\nCompleted {total_turns} turns in {elapsed:.1f}s ({elapsed/total_turns:.2f}s/turn)")

    # Phase 2: Test recall
    print("\n" + "-" * 70)
    print("PHASE 2: Testing recall of planted facts...")
    print("-" * 70 + "\n")

    hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))
    recalls_successful = 0

    for i, query in enumerate(scenario.recall_queries):
        print(f"Query {i+1}: {query}")

        result = manager.respond_catalytic(
            query=query,
            llm_generate=lambda s, p: generate_response(s, p),
            system_prompt=system_prompt,
        )

        # Check if the relevant fact is in context
        context_text = " ".join([item.content for item in result.prepare_result.working_set])
        original_fact = scenario.facts[i][1]

        # Simple keyword check
        key_numbers = [s for s in original_fact.split() if any(c.isdigit() for c in s)]
        found = sum(1 for n in key_numbers if n in context_text)
        success = found >= 1 or original_fact[:30].lower() in context_text.lower()

        if success:
            recalls_successful += 1
            print(f"  -> RECALLED (hydrated: {len(result.prepare_result.hydrated_turns)})")
        else:
            print(f"  -> MISSED")

        print(f"  Response: {result.response[:150]}...")
        print()

    hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

    # Summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    stats = manager.get_compression_stats()
    state = manager.context_state

    print(f"\nRecall Performance:")
    print(f"  Facts recalled: {recalls_successful}/{len(scenario.recall_queries)}")
    print(f"  Hydrations triggered: {hydration_after - hydration_before}")

    print(f"\nCompression Stats:")
    print(f"  Turns compressed: {stats['turns_compressed']}")
    print(f"  Original tokens: {stats['total_original_tokens']}")
    print(f"  Pointer tokens: {stats['total_pointer_tokens']}")
    print(f"  Tokens saved: {stats.get('tokens_saved', 0)}")
    print(f"  Compression ratio: {stats['average_compression_ratio']:.2f}x")

    print(f"\nContext State:")
    print(f"  Working set: {len(state.working_set)} items")
    print(f"  Pointer set: {len(state.pointer_set)} items")
    print(f"  Turn pointers: {len(state.turn_pointers)}")
    print(f"  Budget utilization: {state.utilization_pct:.1%}")

    # Verify chain
    is_valid, error = capsule.verify_chain(session_id)
    print(f"\nChain integrity: {'VALID' if is_valid else f'INVALID - {error}'}")

    capsule.close()

    print("\n" + "=" * 70)
    if recalls_successful == len(scenario.recall_queries):
        print("SUCCESS: All facts recalled after compression!")
    elif recalls_successful > 0:
        print(f"PARTIAL: {recalls_successful}/{len(scenario.recall_queries)} facts recalled")
    else:
        print("FAILED: No facts recalled - investigate rehydration")
    print("=" * 70)

    return recalls_successful, len(scenario.recall_queries)


def run_interactive():
    """Run an interactive demo session."""
    import tempfile

    print("\n" + "=" * 70)
    print("CATALYTIC CONTEXT: Interactive Demo")
    print("=" * 70)
    print("Commands: 'stats' (show stats), 'events' (show events), 'quit' (exit)")
    print("=" * 70 + "\n")

    # Setup
    tmpdir = Path(tempfile.gettempdir()) / "catalytic_interactive"
    tmpdir.mkdir(exist_ok=True)
    db_path = tmpdir / "interactive.db"

    if db_path.exists():
        db_path.unlink()

    capsule = SessionCapsule(db_path=db_path)
    session_id = capsule.create_session()

    system_prompt = "You are a helpful assistant with perfect memory."
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=CONTEXT_WINDOW,
        system_prompt=system_prompt,
        response_reserve_pct=0.25,
        model_id=NEMOTRON_MODEL,
    )

    manager = AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=get_embedding,
        E_threshold=0.3,
    )
    manager.capsule = capsule

    turn = 0
    while True:
        try:
            query = input(f"\n[Turn {turn+1}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == 'quit':
            break
        if query.lower() == 'stats':
            stats = manager.get_compression_stats()
            state = manager.context_state
            print(f"\nCompression: {stats['turns_compressed']} turns, {stats['average_compression_ratio']:.2f}x ratio")
            print(f"Working set: {len(state.working_set)} items, {state.utilization_pct:.1%} budget used")
            print(f"Turn pointers: {len(state.turn_pointers)}")
            continue
        if query.lower() == 'events':
            events = capsule.get_events(session_id)
            types = {}
            for e in events:
                types[e.event_type] = types.get(e.event_type, 0) + 1
            print(f"\nEvents: {len(events)} total")
            for t, c in sorted(types.items()):
                print(f"  {t}: {c}")
            continue

        turn += 1
        start = time.time()

        result = manager.respond_catalytic(
            query=query,
            llm_generate=lambda s, p: generate_response(s, p),
            system_prompt=system_prompt,
        )

        elapsed = time.time() - start

        print(f"\nAssistant: {result.response}")
        print(f"\n[E_mean={result.E_mean:.3f}, hydrated={len(result.prepare_result.hydrated_turns)}, " \
              f"compression={result.compression_ratio:.1f}x, time={elapsed:.1f}s]")

    capsule.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Catalytic Context Demo with Nemotron")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="physics",
                        help="Demo scenario to run")
    parser.add_argument("--turns", type=int, default=50,
                        help="Total conversation turns")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive chat instead of scripted demo")
    parser.add_argument("--all", action="store_true",
                        help="Run all scenarios")
    args = parser.parse_args()

    # Check LLM Studio
    available, msg = check_llm_studio()
    if not available:
        print(f"Error: LLM Studio not available - {msg}")
        print(f"Make sure LLM Studio is running at {LLM_STUDIO_BASE}")
        sys.exit(1)

    print(f"LLM Studio: {msg}")

    if args.interactive:
        run_interactive()
    elif args.all:
        results = []
        for name, scenario in SCENARIOS.items():
            recalled, total = run_demo(scenario, args.turns)
            results.append((name, recalled, total))
            print("\n")

        print("\n" + "=" * 70)
        print("ALL SCENARIOS SUMMARY")
        print("=" * 70)
        for name, recalled, total in results:
            pct = recalled / total * 100
            bar = "#" * int(pct / 5)
            print(f"  {name:30s}: {recalled}/{total} ({pct:.0f}%) {bar}")
    else:
        scenario = SCENARIOS[args.scenario]
        run_demo(scenario, args.turns, verbose=True)


if __name__ == "__main__":
    main()
