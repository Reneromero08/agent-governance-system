#!/usr/bin/env python3
"""Simple logic test - just show responses."""

import sys
from pathlib import Path
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve_with_scores


def ask_model(query: str, context: list, model: str) -> str:
    """Ask model with optional context."""
    context_block = ""
    if context:
        context_block = "\n\n--- CONTEXT (Use this to solve the problem) ---\n"
        for i, doc in enumerate(context, 1):
            context_block += f"\n[{i}] {doc}\n"
        context_block += "\n--- END CONTEXT ---\n\n"

    prompt = f"""{context_block}{query}

Answer concisely with just the solution."""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 300}
    )
    return result['response'].strip()


query = "On an island, knights always tell truth and knaves always lie. You meet two people. A says 'We are both knaves'. What are A and B? Answer format: A is X, B is Y"

print("="*60)
print("LOGIC PROBLEM")
print("="*60)
print(f"Query: {query}\n")

# Get context
context_results = retrieve_with_scores(query, k=4, threshold=0.2, domain='logic')
context = [content for E, Df, content in context_results]

print(f"Retrieved {len(context)} docs")
for i, (E, Df, content) in enumerate(context_results, 1):
    preview = content[:80].replace('\n', ' ')
    print(f"  [{i}] E={E:.3f}: {preview}...")
print()

# Big without context
print("[1] BIG (7b) - NO CONTEXT")
response = ask_model(query, [], "qwen2.5-coder:7b")
print(f"{response}\n")

# Tiny without context (3b)
print("[2] TINY (3b) - NO CONTEXT")
response = ask_model(query, [], "qwen2.5-coder:3b")
print(f"{response}\n")

# Tiny WITH context (3b)
print("[3] TINY (3b) - WITH CONTEXT (QUANTUM RESCUE)")
response = ask_model(query, context, "qwen2.5-coder:3b")
print(f"{response}\n")

# Super tiny without context (0.5b)
print("[4] SUPER TINY (0.5b) - NO CONTEXT")
response = ask_model(query, [], "qwen2.5-coder:0.5b")
print(f"{response}\n")

# Super tiny WITH context (0.5b)
print("[5] SUPER TINY (0.5b) - WITH CONTEXT")
response = ask_model(query, context, "qwen2.5-coder:0.5b")
print(f"{response}\n")

print("="*60)
print("Correct answer: A is knave, B is knight")
print("="*60)
