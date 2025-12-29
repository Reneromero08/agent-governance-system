#!/usr/bin/env python3
"""
Semantic Network Hub - Central coordinator for cassette network.

Manages all cassettes and routes queries based on capabilities.
"""

from typing import Dict, List
from cassette_protocol import DatabaseCassette


class SemanticNetworkHub:
    """Central hub for cassette network.

    Coordinates query routing across multiple cassettes.
    """

    def __init__(self):
        self.cassettes: Dict[str, DatabaseCassette] = {}
        self.protocol_version = "1.0"

    def register_cassette(self, cassette: DatabaseCassette) -> Dict:
        """Register a new cassette in network.

        Args:
            cassette: DatabaseCassette instance

        Returns:
            Handshake metadata from cassette
        """
        handshake = cassette.handshake()
        self.cassettes[handshake['cassette_id']] = cassette
        print(f"[NETWORK] Registered cassette: {handshake['cassette_id']}")
        print(f"  Path: {handshake['db_path']}")
        print(f"  Hash: {handshake['db_hash']}")
        print(f"  Capabilities: {handshake['capabilities']}")
        print(f"  Stats: {handshake['stats']}")
        return handshake

    def query_all(self, query: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query all cassettes and aggregate results.

        Args:
            query: Search query string
            top_k: Results per cassette

        Returns:
            Dict mapping cassette_id to list of results
        """
        print(f"[NETWORK] Routing query to all cassettes: '{query}'")

        results = {}
        for cassette_id, cassette in self.cassettes.items():
            try:
                cassette_results = cassette.query(query, top_k)
                results[cassette_id] = cassette_results
                print(f"  {cassette_id}: {len(cassette_results)} results")
            except Exception as e:
                print(f"  {cassette_id}: ERROR - {e}")
                results[cassette_id] = []

        return results

    def query_by_capability(self, query: str, capability: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query only cassettes with specific capability.

        Args:
            query: Search query string
            capability: Required capability (e.g., 'vectors', 'ast', 'research')
            top_k: Results per cassette

        Returns:
            Dict mapping cassette_id to list of results
        """
        print(f"[NETWORK] Routing query to cassettes with capability '{capability}': '{query}'")

        results = {}
        for cassette_id, cassette in self.cassettes.items():
            if capability in cassette.capabilities:
                try:
                    cassette_results = cassette.query(query, top_k)
                    results[cassette_id] = cassette_results
                    print(f"  {cassette_id}: {len(cassette_results)} results")
                except Exception as e:
                    print(f"  {cassette_id}: ERROR - {e}")
                    results[cassette_id] = []

        return results

    def get_network_status(self) -> Dict:
        """Get status of all registered cassettes."""
        status = {
            "protocol_version": self.protocol_version,
            "cassette_count": len(self.cassettes),
            "cassettes": {}
        }

        for cassette_id, cassette in self.cassettes.items():
            status["cassettes"][cassette_id] = {
                "db_path": str(cassette.db_path),
                "db_exists": cassette.db_path.exists(),
                "capabilities": cassette.capabilities,
                "stats": cassette.get_stats()
            }

        return status

    def print_network_status(self):
        """Print formatted network status."""
        print("\n" + "=" * 70)
        print("CASSETTE NETWORK STATUS")
        print("=" * 70)
        print(f"Protocol Version: {self.protocol_version}")
        print(f"Registered Cassettes: {len(self.cassettes)}\n")

        for cassette_id, cassette in self.cassettes.items():
            stats = cassette.get_stats()
            print(f"Cassette: {cassette_id}")
            print(f"  Database: {cassette.db_path.name}")
            print(f"  Exists: {cassette.db_path.exists()}")
            print(f"  Capabilities: {cassette.capabilities}")
            print(f"  Stats: {stats}")
            print()
