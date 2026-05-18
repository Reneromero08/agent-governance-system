"""Test Lil Q retrieve.py E-gating against test_sandbox.db"""
import sys, os
os.chdir(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\LIL_Q\test_sandbox")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system")
from retrieve import retrieve_with_scores

queries = [
    "Solve for x: (2x + 3)^2 - (x - 1)^2 = 45",
    "Fix fibonacci recursion with memoization or caching",
    "Knights and knaves logic puzzle truth tables",
    "Balance Fe + O2 yields Fe2O3 chemical equation",
]

for q in queries:
    results = retrieve_with_scores(q, k=3, threshold=0.3)
    print("Query: {}...".format(q[:60]))
    if results:
        for i, (E, Df, content) in enumerate(results, 1):
            print("  [{}] E={:.3f} Df={:.1f}  {}...".format(i, E, Df, content[:100]))
    else:
        print("  [NO RESULTS]")
    print()
