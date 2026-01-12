#!/usr/bin/env python3
"""Quick single test for debugging."""

from retrieve import retrieve_with_scores

# Test math domain retrieval
query = "Solve for x: (2x + 3)^2 - (x - 1)^2 = 45"
print(f"Testing retrieval for query: {query}\n")

results = retrieve_with_scores(query, k=3, threshold=0.3, domain='math')

print(f"\n{'='*60}")
print(f"Final results: {len(results)} documents")
for i, (E, Df, content) in enumerate(results, 1):
    preview = content[:150].replace('\n', ' ')
    print(f"\n[{i}] E={E:.3f}, Df={Df:.1f}")
    print(f"    {preview}...")
