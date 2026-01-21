"""
Demo: Catalytic Chat with Local LLM
===================================

This script demonstrates the auto-controlled context loop with a real local LLM.

Requirements:
- sentence-transformers (for embeddings)
- ollama running locally with a model (e.g., qwen2:7b)

Usage:
    python demo_catalytic_with_local_llm.py

Or with a specific model:
    python demo_catalytic_with_local_llm.py --model qwen2:7b
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

import numpy as np
from datetime import datetime


def check_dependencies():
    """Check required dependencies are available."""
    missing = []

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import requests
        # Check ollama is running
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code != 200:
                missing.append("ollama (not responding)")
        except:
            missing.append("ollama (not running - start with 'ollama serve')")
    except ImportError:
        missing.append("requests")

    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall with: pip install sentence-transformers requests")
        print("Start ollama with: ollama serve")
        return False
    return True


def create_ollama_generator(model: str = "qwen2:7b"):
    """Create LLM generator using Ollama."""
    import requests

    def generate(system_prompt: str, user_prompt: str) -> str:
        """Generate response using Ollama API."""
        payload = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            }
        }

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"[Error: {e}]"

    return generate


def create_embedding_fn():
    """Create embedding function using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(text: str) -> np.ndarray:
        return model.encode(text, convert_to_numpy=True)

    return embed


def run_demo(model_name: str, context_window: int):
    """Run the catalytic chat demo."""
    import tempfile

    from catalytic_chat.session_capsule import SessionCapsule
    from catalytic_chat.auto_context_manager import AutoContextManager
    from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
    from catalytic_chat.context_partitioner import ContextItem

    print("\n" + "=" * 60)
    print("CATALYTIC CHAT - Auto-Controlled Context Loop Demo")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Context Window: {context_window} tokens")
    print("=" * 60 + "\n")

    # Create temp database (use explicit path to avoid Windows cleanup issues)
    tmpdir = Path(tempfile.gettempdir()) / "catalytic_demo"
    tmpdir.mkdir(exist_ok=True)
    db_path = tmpdir / "demo_catalytic.db"

    # Clean up old db if exists
    if db_path.exists():
        try:
            db_path.unlink()
        except:
            pass

    # Initialize components
    print("Initializing...")
    embed_fn = create_embedding_fn()
    llm_generate = create_ollama_generator(model_name)

    # Create session
    capsule = SessionCapsule(db_path=db_path)
    session_id = capsule.create_session()
    print(f"Session: {session_id}")

    # Create budget from model config
    system_prompt = "You are a helpful AI assistant. Answer questions clearly and concisely."
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=context_window,
        system_prompt=system_prompt,
        response_reserve_pct=0.25,
        model_id=model_name,
    )
    print(f"Budget: {budget.available_for_working_set} tokens for working set")

    # Create auto context manager
    manager = AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=embed_fn,
        E_threshold=0.3,  # Lower threshold for demo
    )
    manager.capsule = capsule  # Use same capsule instance

    # Add some initial context documents
    docs = [
        ("catalytic_computing",
         "Catalytic computing is a paradigm where large auxiliary state (catalytic space) "
         "must restore exactly after use. The key insight is that we can use vast resources "
         "during computation as long as we clean up perfectly afterward."),

        ("e_score_born_rule",
         "E-scores measure relevance using the Born rule from quantum mechanics. "
         "E = |<query|item>|^2 gives the probability-like relevance score. "
         "The validated threshold is 0.5, meaning items with E >= 0.5 are considered relevant."),

        ("context_partitioning",
         "Context partitioning divides items into working_set (high-E, in context) and "
         "pointer_set (low-E or over budget, compressed). Every turn re-partitions based "
         "on the current query, ensuring context is always relevant."),

        ("weather_unrelated",
         "The weather forecast shows sunny skies with temperatures around 72F. "
         "Expect clear conditions through the weekend with light winds from the west."),
    ]

    for doc_id, content in docs:
        embedding = embed_fn(content)
        manager.add_item(ContextItem(
            item_id=doc_id,
            content=content,
            tokens=len(content) // 4,
            embedding=embedding,
            item_type="document",
        ))

    print(f"Added {len(docs)} context documents\n")

    # Interactive chat loop
    print("-" * 60)
    print("Chat started! Type 'quit' to exit, 'stats' for metrics.")
    print("-" * 60 + "\n")

    queries = [
        "What is catalytic computing?",
        "How do E-scores work?",
        "What's the weather like?",
        "Explain context partitioning.",
    ]

    for i, query in enumerate(queries):
        print(f"\n[Turn {i+1}] User: {query}")
        print("-" * 40)

        # Run catalytic response
        start = datetime.now()
        result = manager.respond_catalytic(
            query=query,
            llm_generate=llm_generate,
            system_prompt=system_prompt,
        )
        elapsed = (datetime.now() - start).total_seconds()

        # Show response
        print(f"Assistant: {result.response[:500]}{'...' if len(result.response) > 500 else ''}")

        # Show metrics
        print(f"\n[Metrics]")
        print(f"  E_mean: {result.E_mean:.3f}")
        print(f"  Working set: {len(result.prepare_result.working_set)} items")
        print(f"  Tokens in context: {result.tokens_in_context}")
        print(f"  Compression ratio: {result.compression_ratio:.1f}x")
        print(f"  Hydrated turns: {len(result.prepare_result.hydrated_turns)}")
        print(f"  Time: {elapsed:.1f}s")
        print("-" * 40)

    # Final stats
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    stats = manager.get_compression_stats()
    print(f"Turns compressed: {stats['turns_compressed']}")
    print(f"Total original tokens: {stats['total_original_tokens']}")
    print(f"Total pointer tokens: {stats['total_pointer_tokens']}")
    print(f"Tokens saved: {stats.get('tokens_saved', 0)}")
    print(f"Average compression ratio: {stats['average_compression_ratio']:.1f}x")

    state = manager.context_state
    print(f"\nFinal context state:")
    print(f"  Working set: {len(state.working_set)} items")
    print(f"  Pointer set: {len(state.pointer_set)} items")
    print(f"  Turn pointers: {len(state.turn_pointers)}")
    print(f"  Budget utilization: {state.utilization_pct:.1%}")

    # Verify events were logged
    events = capsule.get_events(session_id)
    event_types = {}
    for e in events:
        event_types[e.event_type] = event_types.get(e.event_type, 0) + 1

    print(f"\nEvents logged: {len(events)} total")
    for etype, count in sorted(event_types.items()):
        print(f"  {etype}: {count}")

    # Verify chain integrity
    is_valid, error = capsule.verify_chain(session_id)
    print(f"\nChain integrity: {'VALID' if is_valid else f'INVALID - {error}'}")

    capsule.close()

    print("\n" + "=" * 60)
    print("Demo complete! The system is catalytic:")
    print("- Every turn compressed immediately")
    print("- High-E turns rehydrated when relevant")
    print("- Budget respected throughout")
    print("- All decisions logged for replay")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Catalytic Chat Demo with Local LLM")
    parser.add_argument("--model", default="qwen2:7b", help="Ollama model name")
    parser.add_argument("--context-window", type=int, default=8192,
                        help="Model context window size")
    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    run_demo(args.model, args.context_window)


if __name__ == "__main__":
    main()
