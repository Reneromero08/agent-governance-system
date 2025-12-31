import sys
sys.path.insert(0, '.')
from NAVIGATION.CORTEX.semantic.semantic_search import search_cortex

print("Testing semantic search with various queries...")

queries = [
    "agent governance",
    "catalytic computing", 
    "vector embeddings",
    "MCP server",
    "@Symbol system"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    try:
        results = search_cortex(query, top_k=3)
        for i, result in enumerate(results, 1):
            file_name = result.file_path.split('/')[-1] if result.file_path else "unknown"
            print(f"{i}. {file_name} (similarity: {result.similarity:.3f})")
            # Show snippet
            snippet = result.content[:80] + "..." if len(result.content) > 80 else result.content
            print(f"   '{snippet}'")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 50)
print("Semantic search test complete!")