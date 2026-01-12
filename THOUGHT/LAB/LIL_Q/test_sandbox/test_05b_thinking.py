#!/usr/bin/env python3
"""Test 0.5b with extended thinking."""

import sys
from pathlib import Path
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve_with_scores


def ask_with_thinking(query: str, context: list, model: str) -> str:
    """Ask model with thinking steps."""
    context_block = ""
    if context:
        context_block = "\n\n--- KNOWLEDGE BASE ---\n"
        for i, doc in enumerate(context, 1):
            context_block += f"\n[DOC {i}]\n{doc}\n"
        context_block += "\n--- END KNOWLEDGE BASE ---\n\n"

    prompt = f"""{context_block}Problem: {query}

Think through this step by step:

Step 1: What does A claim?
Step 2: If A is a knight (truth-teller), what does that mean?
Step 3: If A is a knave (liar), what does that mean?
Step 4: Which scenario is consistent?

Now give your final answer in the format: A is X, B is Y"""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 500}
    )
    return result['response'].strip()


query = "On an island, knights always tell truth and knaves always lie. You meet two people. A says 'We are both knaves'. What are A and B?"

print("="*60)
print("0.5B MODEL WITH EXTENDED THINKING")
print("="*60)

# Get context
context_results = retrieve_with_scores(query, k=4, threshold=0.2, domain='logic')
context = [content for E, Df, content in context_results]

print(f"\nRetrieved {len(context)} docs\n")

# Test with thinking mode
print("[1] 0.5b WITH CONTEXT + THINKING MODE")
response = ask_with_thinking(query, context, "qwen2.5-coder:0.5b")
print(f"{response}\n")

print("="*60)
print("Correct: A is knave, B is knight")
print("="*60)
