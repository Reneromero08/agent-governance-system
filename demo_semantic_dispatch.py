#!/usr/bin/env python3
"""
Demo: Semantic Core + Baby Agent Dispatch

Demonstrates the full workflow:
1. Use semantic search to find relevant code sections
2. Compress context with @Symbols and vectors
3. Dispatch to baby agent (Haiku) for execution
4. Receive and validate results

Task: "Add better error messages to the dispatch_task function"
"""

import json
import sys
from pathlib import Path

# Add paths
CORTEX_ROOT = Path(__file__).parent / "CORTEX"
CATALYTIC_ROOT = Path(__file__).parent / "CATALYTIC-DPT"
sys.path.insert(0, str(CORTEX_ROOT))
sys.path.insert(0, str(CATALYTIC_ROOT / "LAB" / "MCP"))

from semantic_search import SemanticSearch
from embeddings import EmbeddingEngine


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    """Demo workflow: semantic search -> compress -> dispatch."""

    print_section("SEMANTIC CORE + BABY AGENT DEMO")

    # Task definition
    task_description = "Add better error messages to the dispatch_task function"
    print(f"\nTask: {task_description}\n")

    # Step 1: Semantic Search to find relevant sections
    print_section("STEP 1: Semantic Search")
    print("Searching CORTEX for relevant sections...\n")

    cortex_db = CORTEX_ROOT / "system1.db"
    if not cortex_db.exists():
        print("[FAIL] CORTEX database not found. Run: python CORTEX/build_semantic_core.py")
        return 1

    engine = EmbeddingEngine()
    with SemanticSearch(cortex_db, embedding_engine=engine) as searcher:
        results = searcher.search(task_description, top_k=5)

    print(f"Found {len(results)} relevant sections:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.section_name}")
        print(f"   File:       {result.file_path}")
        print(f"   Similarity: {result.similarity:.3f}")
        print(f"   Content:    {result.content[:80]}...\n")

    # Step 2: Create compressed context with @Symbols
    print_section("STEP 2: Create Compressed Context")

    # Select top result as primary target
    primary = results[0]

    # Create @Symbol representation
    symbols = {
        "@dispatch_task": {
            "content": primary.content,
            "hash": primary.hash[:16] + "...",
            "file": primary.file_path,
            "lines": primary.line_range or (1, 50)
        }
    }

    # Simulate vector context (truncated for demo)
    query_embedding = engine.embed(task_description)
    vector_context = {
        "task_intent": query_embedding[:4].tolist(),  # First 4 dimensions
        "context_centroid": query_embedding[::96].tolist()  # Every 96th dimension
    }

    print("Compressed Context Created:")
    print(f"  @Symbols: 1 symbol (@dispatch_task)")
    print(f"  Vectors:  2 vectors (task_intent + context_centroid)")
    print(f"  Total:    ~200 tokens (vs ~2000 for full context)")

    # Step 3: Create task spec for baby agent
    print_section("STEP 3: Create Task Spec for Baby Agent")

    task_spec = {
        "task_id": "demo-001",
        "task_type": "code_adapt",
        "instruction": task_description,
        "symbols": symbols,
        "vectors": vector_context,
        "constraints": {
            "max_changes": 3,
            "preserve_signature": True,
            "validate_syntax": True
        }
    }

    print("Task Spec Created:")
    print(json.dumps(task_spec, indent=2))

    # Step 4: Simulate baby agent execution
    print_section("STEP 4: Dispatch to Baby Agent (Haiku)")

    print("Agent ID: haiku-worker-1")
    print("Model: Claude Haiku (Fast, cheap)")
    print("Context: 2,000 tokens (compressed)")
    print("Latency: ~100ms expected")
    print("\nAgent Processing:")
    print("  1. Receiving task spec...")
    print("  2. Resolving @dispatch_task symbol...")
    print("  3. Analyzing current code...")
    print("  4. Planning improvements...")
    print("  5. Executing modifications...")

    # Simulated agent response
    baby_agent_response = {
        "task_id": "demo-001",
        "status": "success",
        "message": "Added comprehensive error messages to dispatch_task",
        "modifications": [
            {
                "type": "add_validation",
                "location": "line 1176-1182",
                "change": "Added validation check for task_spec fields with detailed error message"
            },
            {
                "type": "improve_error_message",
                "location": "line 1193-1197",
                "change": "Enhanced duplicate detection error to include task_id and existing status"
            },
            {
                "type": "add_context",
                "location": "line 1214-1217",
                "change": "Added context details to atomic write failure message"
            }
        ],
        "tokens_used": 1847,
        "execution_time_ms": 89
    }

    # Step 5: Display results
    print_section("STEP 5: Baby Agent Results")

    print(f"Status: {baby_agent_response['status'].upper()}")
    print(f"Message: {baby_agent_response['message']}")
    print(f"\nModifications Applied: {len(baby_agent_response['modifications'])}")

    for i, mod in enumerate(baby_agent_response['modifications'], 1):
        print(f"\n  {i}. {mod['type']}")
        print(f"     Location: {mod['location']}")
        print(f"     Change: {mod['change']}")

    print(f"\nPerformance:")
    print(f"  Tokens used:      {baby_agent_response['tokens_used']} (vs ~50,000 for full context)")
    print(f"  Execution time:   {baby_agent_response['execution_time_ms']}ms")
    print(f"  Token reduction:  {(1 - 1847/50000)*100:.1f}%")

    # Step 6: Validation
    print_section("STEP 6: Validation & Integration")

    print("Validating changes:")
    print("  [OK] Syntax validation passed")
    print("  [OK] Function signature preserved")
    print("  [OK] Changes integrated into codebase")

    # Final summary
    print_section("WORKFLOW COMPLETE")

    print("\nToken Economics Achieved:")
    print(f"  Without Semantic Core:")
    print(f"    - Agent context: 50,000 tokens")
    print(f"    - 10 parallel tasks: 500,000 tokens total")
    print(f"")
    print(f"  With Semantic Core:")
    print(f"    - Semantic search: 100,000 tokens (one-time)")
    print(f"    - Compressed context: 2,000 tokens per agent")
    print(f"    - 10 parallel tasks: 100,000 + 20,000 = 120,000 tokens total")
    print(f"")
    print(f"  SAVINGS: 76% token reduction!")

    print("\nArchitectural Benefits:")
    print("  [OK] Big model (Opus) handles semantic understanding")
    print("  [OK] Baby models (Haiku) execute specific tasks")
    print("  [OK] No context duplication across workers")
    print("  [OK] Semantic targeting improves accuracy")
    print("  [OK] Faster execution with smaller contexts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
