#!/usr/bin/env python3
"""Test just the math problem."""

import sys
from pathlib import Path
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve_with_scores


def ask_model(query: str, context: list, model: str) -> str:
    """Ask model with optional context."""
    # Build prompt with context
    context_block = ""
    if context:
        context_block = "\n\n--- CONTEXT ---\n"
        for i, doc in enumerate(context, 1):
            preview = doc[:800] if len(doc) > 800 else doc
            context_block += f"\n[{i}] {preview}\n"
        context_block += "\n--- END CONTEXT ---\n\n"

    prompt = f"""{context_block}Solve this problem step by step:

{query}"""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 500}
    )
    return result['response'].strip()


# Math problem
query = "Solve for x: (2x + 3)^2 - (x - 1)^2 = 45"

print("="*60)
print("MATH PROBLEM TEST")
print("="*60)

# Retrieve context
print("\n[1] Retrieving context...")
context_results = retrieve_with_scores(query, k=3, threshold=0.3, domain='math')
context = [content for E, Df, content in context_results]

print(f"Retrieved {len(context)} documents")
for i, (E, Df, _) in enumerate(context_results, 1):
    print(f"  Doc {i}: E={E:.3f}, Df={Df:.1f}")

# Test big model WITHOUT context
print("\n[2] Testing BIG model (7b) WITHOUT context...")
big_no_ctx = ask_model(query, [], "qwen2.5-coder:7b")
print(f"Response:\n{big_no_ctx}\n")

# Test tiny model WITHOUT context
print("\n[3] Testing TINY model (0.5b) WITHOUT context...")
tiny_no_ctx = ask_model(query, [], "qwen2.5-coder:0.5b")
print(f"Response:\n{tiny_no_ctx}\n")

# Test tiny model WITH context (QUANTUM RESCUE!)
print("\n[4] Testing TINY model (0.5b) WITH context (QUANTUM RESCUE)...")
tiny_with_ctx = ask_model(query, context, "qwen2.5-coder:0.5b")
print(f"Response:\n{tiny_with_ctx}\n")

print("\n" + "="*60)
print("DONE")
print("="*60)
