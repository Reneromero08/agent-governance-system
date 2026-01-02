#!/usr/bin/env python3
"""
Cassette Network Demo - Demonstrates AGS + AGI cassette network.

Phase 0: Proof of Concept
Tests cross-database queries between AGS governance and AGI research.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from network_hub import SemanticNetworkHub
from cassettes import GovernanceCassette, AResearchCassette


def main():
    print("=" * 70)
    print("CASSETTE NETWORK DEMO - AGS + AGI")
    print("=" * 70)
    print()

    hub = SemanticNetworkHub()

    print("--- REGISTERING CASSETTES ---")
    print()

    gov_cassette = GovernanceCassette()
    hub.register_cassette(gov_cassette)

    print()

    agi_cassette = AResearchCassette()
    hub.register_cassette(agi_cassette)

    print()

    hub.print_network_status()

    queries = [
        "governance",
        "memory",
        "architecture"
    ]

    print("\n--- CROSS-CASSETTE QUERIES ---")
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        results = hub.query_all(query, top_k=5)

        for cassette_id, cassette_results in results.items():
            print(f"\n{cassette_id} ({len(cassette_results)} results):")
            for i, r in enumerate(cassette_results[:3], 1):
                heading = r.get('heading', r.get('path', 'unknown'))
                content = r['content'][:80] + "..." if len(r['content']) > 80 else r['content']
                print(f"  {i}. [{r['source']}] {heading}")
                print(f"      {content}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
