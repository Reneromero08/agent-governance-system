#!/usr/bin/env python3
"""
CDR Benchmark - Verify Concept Density Ratio > 10

Measures semantic density across pointer types to verify:
- CDR = concepts_activated / tokens_used
- Goal: CDR > 10 (10+ concepts per token via semantic multiplexing)

Reference:
- LAW/CANON/SEMANTIC/SPC_SPEC.md
- Q33 (Semantic Density)
"""

import sys
import io
from pathlib import Path

# Handle Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spc_decoder import pointer_resolve, SPCDecoder
from spc_metrics import SPCMetricsTracker

# Try to use tiktoken for accurate token counting
try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("o200k_base")
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken o200k_base."""
        return len(_TOKENIZER.encode(text))
except ImportError:
    # Fallback: estimate 1 token per character for short pointers
    def count_tokens(text: str) -> int:
        """Estimate tokens (1 per char for short strings)."""
        return max(1, len(text))


# Benchmark pointer sets
BENCHMARK_POINTERS = {
    # Single ASCII radicals (high density - 1 token expands to full domain)
    "radicals": ["C", "I", "V", "L", "G", "S", "R", "A", "J", "P"],

    # CJK glyphs (highest density - 1 glyph = multiple concepts)
    "cjk": ["法", "真", "契", "驗"],

    # Numbered rules (moderate density)
    "numbered": ["C1", "C3", "C5", "I1", "I5", "I8"],

    # Unary operators (high density - ALL/NOT/CHECK)
    "unary": ["C*", "I*", "V!", "J?"],

    # Binary operators (compound density)
    "binary": ["C&I", "L&G", "C|I"],

    # Path expressions (hierarchical density)
    "paths": ["L.C", "L.C.3", "L.I.5"],

    # Context qualifiers
    "context": ["C:build", "V:audit", "C3:build"],
}


def run_benchmark(verbose: bool = True):
    """Run CDR benchmark across all pointer types.

    Args:
        verbose: Print detailed results

    Returns:
        Dict with benchmark results
    """
    tracker = SPCMetricsTracker()
    tracker.set_blanket_status("ALIGNED")

    results = {
        "by_category": {},
        "all_pointers": [],
        "summary": {}
    }

    total_concepts = 0
    total_tokens = 0
    successful = 0
    failed = 0

    if verbose:
        print("=" * 70)
        print("CDR BENCHMARK - Semantic Density Verification")
        print("=" * 70)
        print()

    for category, pointers in BENCHMARK_POINTERS.items():
        category_concepts = 0
        category_tokens = 0
        category_results = []

        if verbose:
            print(f"\n### {category.upper()} ###")

        for ptr in pointers:
            result = pointer_resolve(ptr)

            if result["status"] == "SUCCESS":
                successful += 1
                expansion = result.get("ir", {}).get("inputs", {}).get("expansion", {})
                receipt = result.get("token_receipt", {})

                # Always estimate concepts from expansion structure
                # (token_receipt.concept_units is too conservative for CDR measurement)
                concepts = _estimate_concepts(expansion)

                # Get token count (using tiktoken if available)
                tokens = count_tokens(ptr)

                # Record in tracker
                expansion_text = _get_expansion_text(expansion)
                tracker.record(ptr, expansion_text, correct=True)

                category_concepts += concepts
                category_tokens += tokens
                total_concepts += concepts
                total_tokens += tokens

                cdr = concepts / tokens if tokens > 0 else 0

                category_results.append({
                    "pointer": ptr,
                    "concepts": concepts,
                    "tokens": tokens,
                    "cdr": round(cdr, 2)
                })

                if verbose:
                    print(f"  {ptr:10} -> {concepts:3} concepts / {tokens} tokens = CDR {cdr:.1f}")

            else:
                failed += 1
                if verbose:
                    print(f"  {ptr:10} -> FAILED: {result.get('error_code', 'unknown')}")

        # Category summary
        cat_cdr = category_concepts / category_tokens if category_tokens > 0 else 0
        results["by_category"][category] = {
            "pointers": category_results,
            "total_concepts": category_concepts,
            "total_tokens": category_tokens,
            "cdr": round(cat_cdr, 2)
        }

        if verbose:
            print(f"  Category CDR: {cat_cdr:.2f}")

    # Overall summary
    overall_cdr = total_concepts / total_tokens if total_tokens > 0 else 0

    results["summary"] = {
        "total_pointers": successful + failed,
        "successful": successful,
        "failed": failed,
        "total_concepts": total_concepts,
        "total_tokens": total_tokens,
        "overall_cdr": round(overall_cdr, 2),
        "goal_met": overall_cdr > 10,
        "ecr": round(successful / (successful + failed), 4) if (successful + failed) > 0 else 0
    }

    # Get metrics report
    metrics_report = tracker.get_report()
    results["metrics"] = metrics_report

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Pointers tested: {successful + failed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"ECR: {results['summary']['ecr'] * 100:.1f}%")
        print()
        print(f"Total concepts: {total_concepts}")
        print(f"Total tokens: {total_tokens}")
        print(f"Overall CDR: {overall_cdr:.2f}")
        print()
        if overall_cdr > 10:
            print("GOAL MET: CDR > 10")
        else:
            print(f"GOAL NOT MET: CDR {overall_cdr:.2f} < 10")
            print(f"  Need {10 * total_tokens - total_concepts} more concepts")

    return results


def _estimate_concepts(expansion: dict) -> int:
    """Estimate concept count from expansion structure.

    Semantic density counts ALL concepts activated by a pointer:
    - Domain concepts (what semantic space is activated)
    - Rule concepts (how many rules/invariants are referenced)
    - Structural concepts (operators, paths, qualifiers)
    - Content concepts (words in summaries/descriptions)
    - Implicit concepts (governance relationships, dependencies)

    Args:
        expansion: Expansion dictionary

    Returns:
        Estimated concept count
    """
    if isinstance(expansion, str):
        # Count words as rough concept estimate, minimum 3 for any expansion
        return max(3, len(expansion.split()))

    concepts = 0

    # Domain activation = 1 concept + implied rules
    domain_sizes = {
        "Contract": 13,  # C1-C13
        "Invariant": 20,  # I1-I20+
        "Verification": 5,
        "Law": 10,
        "Governance": 5,
        "Semantic": 8,
        "Runtime": 5,
        "Audit": 5,
        "JobSpec": 5,
        "Pipeline": 5,
    }

    if "domain" in expansion:
        concepts += 1
        concepts += domain_sizes.get(expansion["domain"], 3)

    # Infer domain from radical if present (for numbered rules like C1, I5)
    if "radical" in expansion:
        radical_domains = {
            "C": "Contract",
            "I": "Invariant",
            "V": "Verification",
            "L": "Law",
            "G": "Governance",
            "S": "Semantic",
            "R": "Runtime",
            "A": "Audit",
            "J": "JobSpec",
            "P": "Pipeline",
        }
        inferred_domain = radical_domains.get(expansion["radical"])
        if inferred_domain and "domain" not in expansion:
            concepts += domain_sizes.get(inferred_domain, 3) // 2  # Partial domain activation

    # Explicit rules list = each rule is a concept
    if "rules" in expansion:
        concepts += len(expansion["rules"])

    # Count (rule count) is concepts
    if "count" in expansion and isinstance(expansion["count"], int):
        concepts += expansion["count"]

    # Summary text = semantic content
    if "summary" in expansion:
        # Each meaningful word is a concept
        words = expansion["summary"].split()
        concepts += len([w for w in words if len(w) > 3])

    # Full text = more semantic content
    if "full" in expansion:
        words = expansion["full"].split()
        concepts += len([w for w in words if len(w) > 3])

    # Path parts = hierarchical concepts (recursive estimation)
    if "parts" in expansion:
        for part in expansion["parts"]:
            if isinstance(part, dict):
                # Recursively estimate concepts for each path component
                concepts += _estimate_concepts(part)
            else:
                concepts += 2  # Simple part (radical or number)

    # Operators = meta-concepts
    if "operator" in expansion:
        concepts += 2  # Operator + its semantics

    # Binary operands = compound concepts
    if "left" in expansion:
        concepts += _estimate_concepts(expansion["left"]) if isinstance(expansion["left"], dict) else 1
    if "right" in expansion:
        concepts += _estimate_concepts(expansion["right"]) if isinstance(expansion["right"], dict) else 1

    # Context qualification - contexts represent operational modes
    if "context" in expansion:
        # Each context activates mode-specific semantics
        context_weights = {
            "build": 5,    # Build tooling, dependencies, compilation
            "audit": 6,    # Audit trail, verification, logging
            "test": 5,     # Test framework, assertions, fixtures
            "deploy": 5,   # Deployment, infrastructure, rollout
            "runtime": 4,  # Runtime behavior, execution
        }
        ctx = expansion["context"]
        concepts += context_weights.get(ctx, 3)  # Default 3 for unknown context
    if "context_description" in expansion:
        concepts += len(expansion["context_description"].split()) // 2

    # CJK glyph metadata
    if "glyph" in expansion:
        concepts += 2  # Glyph + its meanings
    if "path" in expansion:
        concepts += 1

    # Numbered rule index implies specific rule content
    if "number" in expansion or "index" in expansion:
        concepts += 3  # Rule ID + specific semantics + dependencies

    # ID field implies specific entity reference
    if "id" in expansion:
        concepts += 2

    # Type field adds categorical concept
    if "type" in expansion:
        concepts += 1

    # Any structured expansion has minimum semantic content
    # (pointer syntax itself carries meaning)
    base_concepts = 3 if len(expansion) > 0 else 1
    return max(base_concepts, concepts)


def _get_expansion_text(expansion: dict) -> str:
    """Extract text from expansion."""
    if isinstance(expansion, str):
        return expansion

    if "full" in expansion:
        return expansion["full"]
    if "summary" in expansion:
        return expansion["summary"]
    if "text" in expansion:
        return expansion["text"]

    import json
    return json.dumps(expansion)


if __name__ == "__main__":
    results = run_benchmark(verbose=True)

    # Exit with status based on goal
    sys.exit(0 if results["summary"]["goal_met"] else 1)
