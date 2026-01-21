#!/usr/bin/env python3
"""
CATALYTIC CONTEXT MARATHON: Liquid Model
=========================================

Tests the paradigm shift with liquid/lfm2.5-1.2b over extended conversations.

Default: 200 turns (practical runtime ~30 min)
Full: 1000 turns (long runtime)

Author: Paradigm Shift Validation
"""

import argparse
import sys
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

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
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"


# =============================================================================
# API Functions
# =============================================================================

def get_available_models() -> List[str]:
    """Get available models from LLM Studio."""
    try:
        resp = requests.get(f"{LLM_STUDIO_BASE}/v1/models", timeout=5)
        if resp.status_code == 200:
            return [m["id"] for m in resp.json().get("data", [])]
    except:
        pass
    return []


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


def generate_response(model: str, system: str, prompt: str, max_tokens: int = 128) -> str:
    """Generate response from LLM."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# =============================================================================
# Fact Generation
# =============================================================================

def generate_fact(fact_id: int, topic: str) -> Tuple[str, List[str]]:
    """Generate a unique fact with keywords for recall testing."""
    facts = {
        "physics": [
            ("FACT-{id}: The quantum coherence time QCT-{id} is {val} microseconds.",
             ["QCT-{id}", "{val}", "microseconds"]),
            ("FACT-{id}: Photon wavelength PWL-{id} measures {val} nanometers.",
             ["PWL-{id}", "{val}", "nanometers"]),
        ],
        "chemistry": [
            ("FACT-{id}: Molecular bond angle MBA-{id} is {val} degrees.",
             ["MBA-{id}", "{val}", "degrees"]),
            ("FACT-{id}: Reaction rate constant RRK-{id} equals {val} per second.",
             ["RRK-{id}", "{val}", "per second"]),
        ],
        "math": [
            ("FACT-{id}: Prime sequence PSQ-{id} starts at {val}.",
             ["PSQ-{id}", "{val}"]),
            ("FACT-{id}: Matrix determinant MDT-{id} equals {val}.",
             ["MDT-{id}", "{val}"]),
        ],
        "astronomy": [
            ("FACT-{id}: Star magnitude SMG-{id} is {val} absolute.",
             ["SMG-{id}", "{val}", "absolute"]),
            ("FACT-{id}: Orbital period ORP-{id} spans {val} days.",
             ["ORP-{id}", "{val}", "days"]),
        ],
        "biology": [
            ("FACT-{id}: Cell count CCN-{id} reached {val} million.",
             ["CCN-{id}", "{val}", "million"]),
            ("FACT-{id}: Protein mass PMX-{id} weighs {val} kilodaltons.",
             ["PMX-{id}", "{val}", "kilodaltons"]),
        ],
    }

    templates = facts.get(topic, facts["physics"])
    template, keywords = templates[fact_id % len(templates)]

    val = 100 + fact_id * 17  # Unique value
    fact = template.format(id=f"{fact_id:04d}", val=val)
    kws = [k.format(id=f"{fact_id:04d}", val=val) for k in keywords]

    return fact, kws


def generate_filler(turn: int) -> str:
    """Generate filler conversation."""
    topics = [
        "weather patterns", "cooking recipes", "sports news",
        "travel destinations", "music genres", "book recommendations",
    ]
    topic = topics[turn % len(topics)]
    return f"Tell me something interesting about {topic}."


# =============================================================================
# Marathon Runner
# =============================================================================

@dataclass
class MarathonResult:
    """Results from marathon run."""
    total_turns: int
    facts_planted: int
    facts_recalled: int
    recall_rate: float
    total_hydrations: int
    compression_ratio: float
    tokens_saved: int
    elapsed_seconds: float
    recall_by_distance: Dict[str, float] = field(default_factory=dict)


def run_marathon(
    model: str,
    total_turns: int = 200,
    context_window: int = 32768,
    verbose: bool = True,
) -> MarathonResult:
    """Run the marathon test with real LLM."""
    import tempfile

    print("\n" + "=" * 70)
    print(f"CATALYTIC MARATHON: {total_turns} Turns with {model}")
    print("=" * 70)

    # Setup
    tmpdir = Path(tempfile.gettempdir()) / "catalytic_marathon"
    tmpdir.mkdir(exist_ok=True)
    db_path = tmpdir / f"marathon_{total_turns}.db"

    if db_path.exists():
        db_path.unlink()

    capsule = SessionCapsule(db_path=db_path)
    session_id = capsule.create_session()

    system_prompt = "You are an assistant with precise memory. Remember facts exactly as given."
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=context_window,
        system_prompt=system_prompt,
        response_reserve_pct=0.25,
        model_id=model,
    )

    manager = AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=get_embedding,
        E_threshold=0.3,
    )
    manager.capsule = capsule

    print(f"Budget: {budget.available_for_working_set} tokens")
    print(f"Session: {session_id}")

    def llm_generate(s: str, p: str) -> str:
        return generate_response(model, s, p)

    # Plan facts at strategic intervals
    topics = ["physics", "chemistry", "math", "astronomy", "biology"]

    # Plant facts at 5%, 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85%, 95% of total
    plant_pcts = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    plant_turns = [max(1, int(total_turns * p / 100)) for p in plant_pcts]

    # Recall checkpoints at 50% and 100%
    recall_checkpoints = [int(total_turns * 0.5), total_turns]

    planted_facts: Dict[int, Tuple[str, List[str], int]] = {}
    recall_attempts: List[Tuple[int, int, bool]] = []  # (fact_id, distance, success)

    start_time = time.time()

    # Phase 1: Run conversation
    print(f"\nPhase 1: Running {total_turns} turns with {len(plant_turns)} planted facts...")

    for turn in range(1, total_turns + 1):
        if turn in plant_turns:
            fact_id = plant_turns.index(turn)
            topic = topics[fact_id % len(topics)]
            fact, keywords = generate_fact(fact_id, topic)
            planted_facts[turn] = (fact, keywords, fact_id)

            try:
                result = manager.respond_catalytic(
                    query=f"Remember this: {fact}",
                    llm_generate=llm_generate,
                    system_prompt=system_prompt,
                )
                if verbose:
                    print(f"  Turn {turn}: Planted FACT-{fact_id:04d} ({topic})")
            except Exception as e:
                print(f"  Turn {turn}: ERROR planting fact - {e}")
        else:
            try:
                manager.respond_catalytic(
                    query=generate_filler(turn),
                    llm_generate=llm_generate,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                print(f"  Turn {turn}: ERROR on filler - {e}")

        # Progress
        if turn % 50 == 0:
            elapsed = time.time() - start_time
            rate = turn / elapsed
            eta = (total_turns - turn) / rate if rate > 0 else 0
            print(f"  Progress: {turn}/{total_turns} ({turn/total_turns:.0%}), "
                  f"{rate:.1f} turns/sec, ETA: {eta/60:.1f} min")

    # Phase 2: Recall testing
    print(f"\nPhase 2: Testing recall...")

    total_recalls = 0
    total_attempts = 0

    for checkpoint in recall_checkpoints:
        print(f"\n  === Checkpoint at turn {checkpoint} ===")
        checkpoint_recalls = 0
        checkpoint_attempts = 0

        for plant_turn, (fact, keywords, fact_id) in planted_facts.items():
            if plant_turn >= checkpoint:
                continue

            total_attempts += 1
            checkpoint_attempts += 1

            try:
                result = manager.respond_catalytic(
                    query=f"What was FACT-{fact_id:04d}?",
                    llm_generate=llm_generate,
                    system_prompt=system_prompt,
                )

                context = " ".join([item.content for item in result.prepare_result.working_set])
                found = sum(1 for kw in keywords if kw in context)
                success = found >= 1

                if success:
                    total_recalls += 1
                    checkpoint_recalls += 1

                distance = checkpoint - plant_turn
                recall_attempts.append((fact_id, distance, success))

            except Exception as e:
                print(f"    ERROR recalling FACT-{fact_id:04d}: {e}")

        print(f"    Recalled: {checkpoint_recalls}/{checkpoint_attempts}")

    elapsed = time.time() - start_time

    # Calculate distance buckets
    distance_buckets = {
        "0-25": [],
        "26-50": [],
        "51-100": [],
        "101-200": [],
        "200+": [],
    }

    for fact_id, distance, success in recall_attempts:
        if distance <= 25:
            distance_buckets["0-25"].append(success)
        elif distance <= 50:
            distance_buckets["26-50"].append(success)
        elif distance <= 100:
            distance_buckets["51-100"].append(success)
        elif distance <= 200:
            distance_buckets["101-200"].append(success)
        else:
            distance_buckets["200+"].append(success)

    recall_by_distance = {}
    for bucket, results in distance_buckets.items():
        if results:
            recall_by_distance[bucket] = sum(results) / len(results)

    # Final stats
    stats = manager.get_compression_stats()
    hydrations = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

    capsule.close()

    result = MarathonResult(
        total_turns=total_turns,
        facts_planted=len(planted_facts),
        facts_recalled=total_recalls,
        recall_rate=total_recalls / total_attempts if total_attempts > 0 else 0,
        total_hydrations=hydrations,
        compression_ratio=stats["average_compression_ratio"],
        tokens_saved=stats.get("tokens_saved", 0),
        elapsed_seconds=elapsed,
        recall_by_distance=recall_by_distance,
    )

    # Print summary
    print("\n" + "=" * 70)
    print(f"MARATHON RESULTS: {total_turns} turns with {model}")
    print("=" * 70)
    print(f"\nExecution time: {elapsed/60:.1f} minutes ({elapsed/total_turns:.2f}s per turn)")
    print(f"\nRecall rate: {result.recall_rate:.1%} ({total_recalls}/{total_attempts})")
    print(f"Total hydrations: {hydrations}")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    print(f"Tokens saved: {result.tokens_saved}")

    print("\nRecall by distance:")
    for bucket, rate in result.recall_by_distance.items():
        bar = "#" * int(rate * 40)
        print(f"  {bucket:>10}: {rate:.1%} {bar}")

    if result.recall_rate >= 0.8:
        print("\n" + "=" * 70)
        print("PARADIGM SHIFT CONFIRMED!")
        print("=" * 70)
    elif result.recall_rate >= 0.6:
        print("\nGood recall maintained.")
    else:
        print("\nRecall degraded - investigate.")

    return result


def main():
    parser = argparse.ArgumentParser(description="Catalytic Marathon with Real LLM")
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detects if not specified)")
    parser.add_argument("--turns", type=int, default=200,
                        help="Total conversation turns (default: 200)")
    parser.add_argument("--context-window", type=int, default=32768,
                        help="Model context window size")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    args = parser.parse_args()

    # Check LLM Studio
    models = get_available_models()
    if not models:
        print("Error: Cannot connect to LLM Studio at", LLM_STUDIO_BASE)
        sys.exit(1)

    print(f"Available models: {models}")

    if args.model is None:
        # Prefer liquid model if available
        liquid = [m for m in models if "lfm" in m.lower() or "liquid" in m.lower()]
        if liquid:
            args.model = liquid[0]
        else:
            args.model = models[0]

    if args.model not in models:
        print(f"Error: Model {args.model} not available")
        sys.exit(1)

    run_marathon(
        model=args.model,
        total_turns=args.turns,
        context_window=args.context_window,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
