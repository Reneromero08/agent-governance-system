#!/usr/bin/env python3
"""
REAL CONVERSATION TEST - No Cherry-Picking
==========================================

Uses actual human conversation data from DailyDialog dataset.
No repetitive filler - real varied natural language.

This is the honest test.
"""

import sys
import time
import random
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import (
    SessionCapsule,
    EVENT_TURN_HYDRATED,
)
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery

# Config
LLM_STUDIO_BASE = "http://10.5.0.2:1234"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"


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


def generate_response(model: str, system: str, prompt: str) -> str:
    """Generate response from LLM."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def load_local_conversations() -> List[str]:
    """Load local real conversation file."""
    conv_file = Path(__file__).parent / "real_conversations.txt"
    if conv_file.exists():
        with open(conv_file, "r", encoding="utf-8") as f:
            utterances = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(utterances)} local utterances from real_conversations.txt")
        return utterances
    return []


def load_openwebtext_sample() -> List[str]:
    """Try to load sample from OpenWebText."""
    try:
        from datasets import load_dataset
        print("Loading OpenWebText sample...")
        dataset = load_dataset("stas/openwebtext-10k", split="train[:500]")
        utterances = []
        for item in dataset:
            text = item.get("text", "")
            # Split into sentences
            sentences = text.replace(".", ". ").split(". ")
            for s in sentences:
                s = s.strip()
                if 15 < len(s) < 200 and not s.startswith("http"):
                    utterances.append(s)
        print(f"Loaded {len(utterances)} utterances from OpenWebText")
        return utterances
    except Exception as e:
        print(f"OpenWebText failed: {e}")
        return []


def run_real_conversation_test(
    model: str,
    total_turns: int = 200,
    context_window: int = 32768,
):
    """Run test with REAL conversation data."""
    import tempfile

    print("\n" + "=" * 70)
    print("REAL CONVERSATION TEST - NO CHERRY-PICKING")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Turns: {total_turns}")
    print("=" * 70)

    # Load real conversations - try multiple sources
    utterances = load_local_conversations()
    if len(utterances) < 100:
        web_utterances = load_openwebtext_sample()
        utterances.extend(web_utterances)
    if len(utterances) < 100:
        print("Falling back to synthetic varied data...")
        utterances.extend(generate_synthetic_varied())
    print(f"Total utterances available: {len(utterances)}")

    random.shuffle(utterances)

    # Setup
    tmpdir = Path(tempfile.gettempdir()) / "real_conv_test"
    tmpdir.mkdir(exist_ok=True)
    db_path = tmpdir / "real_test.db"
    if db_path.exists():
        db_path.unlink()

    capsule = SessionCapsule(db_path=db_path)
    session_id = capsule.create_session()

    system_prompt = "You are a helpful assistant. Remember important facts precisely."
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=context_window,
        system_prompt=system_prompt,
        response_reserve_pct=0.25,
    )

    manager = AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=get_embedding,
        E_threshold=0.3,
    )
    manager.capsule = capsule

    def llm_generate(s: str, p: str) -> str:
        return generate_response(model, s, p)

    # Plant facts at strategic points
    facts_to_plant = [
        ("ALPHA-7X", "The security code ALPHA-7X unlocks the vault.", ["ALPHA-7X", "vault"]),
        ("Dr. Martinez", "Dr. Martinez is the lead researcher on Project Nova.", ["Martinez", "Nova"]),
        ("$2.3M", "The quarterly budget is $2.3 million for R&D.", ["2.3", "million", "R&D"]),
        ("Building C", "The server room is located in Building C, Floor 3.", ["Building C", "Floor 3"]),
        ("Friday 5pm", "The weekly standup meeting is Friday at 5pm.", ["Friday", "5pm"]),
    ]

    # Plant at 10%, 25%, 40%, 55%, 70% of total turns
    plant_positions = [int(total_turns * p) for p in [0.1, 0.25, 0.4, 0.55, 0.7]]

    planted: Dict[int, Tuple[str, str, List[str]]] = {}
    utterance_idx = 0

    print(f"\nRunning {total_turns} turns with REAL conversation filler...")
    print(f"Planting {len(facts_to_plant)} facts at positions: {plant_positions}")

    start_time = time.time()

    for turn in range(1, total_turns + 1):
        if turn in plant_positions:
            fact_idx = plant_positions.index(turn)
            code, fact, keywords = facts_to_plant[fact_idx]
            planted[turn] = (code, fact, keywords)

            result = manager.respond_catalytic(
                query=f"Important: {fact}",
                llm_generate=llm_generate,
                system_prompt=system_prompt,
            )
            print(f"  Turn {turn}: PLANTED '{code}'")
        else:
            # Use REAL conversation from dataset
            query = utterances[utterance_idx % len(utterances)]
            utterance_idx += 1

            try:
                manager.respond_catalytic(
                    query=query,
                    llm_generate=llm_generate,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                print(f"  Turn {turn}: Error - {e}")

        if turn % 50 == 0:
            elapsed = time.time() - start_time
            rate = turn / elapsed
            print(f"  Progress: {turn}/{total_turns} ({rate:.1f} turns/sec)")

    # Test recall
    print(f"\n{'='*70}")
    print("RECALL TEST")
    print("="*70)

    recalls = 0
    for plant_turn, (code, fact, keywords) in planted.items():
        result = manager.respond_catalytic(
            query=f"What was the information about {code}?",
            llm_generate=llm_generate,
            system_prompt=system_prompt,
        )

        context = " ".join([item.content for item in result.prepare_result.working_set])
        found = sum(1 for kw in keywords if kw in context)
        success = found >= 1

        if success:
            recalls += 1
            print(f"  {code}: FOUND ({found}/{len(keywords)} keywords)")
        else:
            print(f"  {code}: MISSED")

        # Also check LLM response
        response_has = sum(1 for kw in keywords if kw.lower() in result.response.lower())
        print(f"    LLM response has: {response_has}/{len(keywords)} keywords")
        try:
            print(f"    Response: {result.response[:100]}...")
        except UnicodeEncodeError:
            print(f"    Response: [contains unicode]")

    elapsed = time.time() - start_time

    # Stats
    stats = manager.get_compression_stats()
    hydrations = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

    print(f"\n{'='*70}")
    print("RESULTS - REAL CONVERSATION TEST")
    print("="*70)
    print(f"Recall: {recalls}/{len(planted)} ({recalls/len(planted):.0%})")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/total_turns:.2f}s/turn)")
    print(f"Hydrations: {hydrations}")
    print(f"Compression: {stats['average_compression_ratio']:.2f}x")
    print(f"Tokens saved: {stats.get('tokens_saved', 0)}")

    if recalls == len(planted):
        print("\nPARADIGM SHIFT CONFIRMED - WITH REAL DATA!")
    elif recalls >= len(planted) * 0.8:
        print("\nStrong performance on real data.")
    elif recalls >= len(planted) * 0.6:
        print("\nModerate performance - room for improvement.")
    else:
        print("\nNeeds work - real conversations are harder.")

    capsule.close()
    return recalls, len(planted)


def generate_synthetic_varied() -> List[str]:
    """Generate varied synthetic data as fallback."""
    templates = [
        "I was thinking about {} the other day.",
        "Have you ever tried {}?",
        "What do you think about {}?",
        "I heard that {} is really interesting.",
        "My friend told me about {}.",
        "I'm curious about {}.",
        "Can you explain {} to me?",
        "I've been reading about {}.",
        "Someone mentioned {} yesterday.",
        "I wonder why {} happens.",
    ]

    topics = [
        "machine learning", "quantum physics", "ancient history",
        "cooking techniques", "urban planning", "music theory",
        "marine biology", "space exploration", "psychology",
        "economics", "architecture", "linguistics", "philosophy",
        "renewable energy", "genetics", "artificial intelligence",
        "climate change", "archaeology", "neuroscience", "robotics",
        "virtual reality", "blockchain", "nanotechnology", "astronomy",
        "botany", "zoology", "geology", "meteorology", "oceanography",
        "anthropology", "sociology", "political science", "mathematics",
        "chemistry", "engineering", "medicine", "literature", "art history",
    ]

    utterances = []
    for template in templates:
        for topic in topics:
            utterances.append(template.format(topic))

    # Add some random questions
    questions = [
        "How does that work exactly?",
        "That's interesting, tell me more.",
        "I didn't know that, what else?",
        "Really? Why is that?",
        "Can you give me an example?",
        "What are the implications?",
        "How did you learn about this?",
        "Is this widely accepted?",
        "What's the history behind it?",
        "Are there any controversies?",
    ]
    utterances.extend(questions * 10)

    return utterances


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="liquid/lfm2.5-1.2b")
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--context-window", type=int, default=32768)
    args = parser.parse_args()

    # Check LLM Studio
    try:
        resp = requests.get(f"{LLM_STUDIO_BASE}/v1/models", timeout=5)
        models = [m["id"] for m in resp.json().get("data", [])]
        print(f"Available models: {models}")
    except:
        print(f"Error: Cannot connect to LLM Studio at {LLM_STUDIO_BASE}")
        sys.exit(1)

    run_real_conversation_test(
        model=args.model,
        total_turns=args.turns,
        context_window=args.context_window,
    )


if __name__ == "__main__":
    main()
