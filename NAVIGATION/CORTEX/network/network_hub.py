#!/usr/bin/env python3
"""
Semantic Network Hub - Central coordinator for cassette network.

Manages all cassettes and routes queries based on capabilities.
Enforces codebook sync before query routing (Phase 4.2).

Reference:
- LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md
- Q35 (Markov Blankets): Blanket alignment gating
"""

import sys
from typing import Dict, List, Optional, Tuple

# Handle both package and direct imports
try:
    from .cassette_protocol import DatabaseCassette
    from .codebook_sync import (
        CodebookSync,
        SyncTuple,
        BlanketStatus,
        SyncErrorCode,
        create_sync_tuple_from_codebook
    )
except ImportError:
    from cassette_protocol import DatabaseCassette
    from codebook_sync import (
        CodebookSync,
        SyncTuple,
        BlanketStatus,
        SyncErrorCode,
        create_sync_tuple_from_codebook
    )


class SemanticNetworkHub:
    """Central hub for cassette network.

    Coordinates query routing across multiple cassettes.
    Enforces codebook synchronization per CODEBOOK_SYNC_PROTOCOL.
    """

    def __init__(
        self,
        verbose: bool = False,
        enforce_sync: bool = True,
        hub_sync_tuple: SyncTuple = None
    ):
        """Initialize network hub.

        Args:
            verbose: Enable verbose logging
            enforce_sync: If True, reject queries to unsynced cassettes
            hub_sync_tuple: Hub's sync tuple (if None, uses first registered cassette's)
        """
        self.cassettes: Dict[str, DatabaseCassette] = {}
        self.protocol_version = "1.0"
        self.verbose = verbose
        self.enforce_sync = enforce_sync

        # Sync state tracking
        self._hub_sync_tuple = hub_sync_tuple
        self._cassette_sync_status: Dict[str, BlanketStatus] = {}
        self._sync_coordinator = CodebookSync(sender_id="network-hub")

    def register_cassette(self, cassette: DatabaseCassette) -> Dict:
        """Register a new cassette in network with sync verification.

        Args:
            cassette: DatabaseCassette instance

        Returns:
            Handshake metadata from cassette including sync status
        """
        handshake = cassette.handshake()
        cassette_id = handshake['cassette_id']

        # Verify sync if enforcement is enabled
        sync_result = self._verify_cassette_sync(handshake)
        handshake['sync_verification'] = sync_result

        # Store cassette and sync status
        self.cassettes[cassette_id] = cassette
        self._cassette_sync_status[cassette_id] = BlanketStatus(
            sync_result.get('blanket_status', 'UNSYNCED')
        )

        if self.verbose:
            print(f"[NETWORK] Registered cassette: {cassette_id}", file=sys.stderr)
            print(f"  Path: {handshake['db_path']}", file=sys.stderr)
            print(f"  Hash: {handshake['db_hash']}", file=sys.stderr)
            print(f"  Capabilities: {handshake['capabilities']}", file=sys.stderr)
            print(f"  Stats: {handshake['stats']}", file=sys.stderr)
            print(f"  Sync Status: {sync_result['blanket_status']}", file=sys.stderr)
            if sync_result.get('mismatches'):
                print(f"  Mismatches: {sync_result['mismatches']}", file=sys.stderr)

        return handshake

    def _verify_cassette_sync(self, handshake: Dict) -> Dict:
        """Verify cassette's sync tuple matches hub's.

        Per CODEBOOK_SYNC_PROTOCOL Section 5.1: Exact match required.

        Args:
            handshake: Cassette handshake data

        Returns:
            Dict with verification result
        """
        cassette_tuple_data = handshake.get('sync_tuple', {})

        # If no sync tuple, cassette is unsynced
        if not cassette_tuple_data:
            return {
                'blanket_status': BlanketStatus.UNSYNCED.value,
                'error': 'No sync_tuple in handshake'
            }

        cassette_tuple = SyncTuple.from_dict(cassette_tuple_data)

        # If hub has no sync tuple yet, adopt first cassette's
        if self._hub_sync_tuple is None:
            self._hub_sync_tuple = cassette_tuple
            return {
                'blanket_status': BlanketStatus.ALIGNED.value,
                'note': 'Hub adopted cassette sync tuple'
            }

        # Check alignment
        is_match, mismatches = self._sync_coordinator.sync_tuples_match(
            cassette_tuple,
            self._hub_sync_tuple
        )

        if is_match:
            # Compute R-value for health tracking
            r_value = self._sync_coordinator.compute_continuous_r(
                cassette_tuple,
                self._hub_sync_tuple
            )
            return {
                'blanket_status': BlanketStatus.ALIGNED.value,
                'r_value': round(r_value, 4)
            }
        else:
            return {
                'blanket_status': BlanketStatus.DISSOLVED.value,
                'mismatches': mismatches,
                'error_code': SyncErrorCode.E_CODEBOOK_MISMATCH.value
            }

    def _is_cassette_synced(self, cassette_id: str) -> bool:
        """Check if cassette is synced and queryable.

        Args:
            cassette_id: Cassette identifier

        Returns:
            True if cassette can receive queries
        """
        status = self._cassette_sync_status.get(cassette_id, BlanketStatus.UNSYNCED)
        return status == BlanketStatus.ALIGNED

    def query_all(self, query: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query all synced cassettes and aggregate results.

        Per CODEBOOK_SYNC_PROTOCOL: Queries only routed to aligned cassettes.

        Args:
            query: Search query string
            top_k: Results per cassette

        Returns:
            Dict mapping cassette_id to list of results (or error dicts)
        """
        if self.verbose:
            print(f"[NETWORK] Routing query to all cassettes: '{query}'", file=sys.stderr)

        results = {}
        for cassette_id, cassette in self.cassettes.items():
            # Check sync status if enforcement is enabled
            if self.enforce_sync and not self._is_cassette_synced(cassette_id):
                status = self._cassette_sync_status.get(cassette_id, BlanketStatus.UNSYNCED)
                if self.verbose:
                    print(f"  {cassette_id}: SKIPPED (blanket {status.value})", file=sys.stderr)
                results[cassette_id] = [{
                    'error': 'E_BLANKET_NOT_ALIGNED',
                    'blanket_status': status.value,
                    'message': 'Cassette sync required before query'
                }]
                continue

            try:
                cassette_results = cassette.query(query, top_k)
                results[cassette_id] = cassette_results
                if self.verbose:
                    print(f"  {cassette_id}: {len(cassette_results)} results", file=sys.stderr)
            except Exception as e:
                print(f"  {cassette_id}: ERROR - {e}", file=sys.stderr)
                results[cassette_id] = []

        return results

    def query_by_capability(self, query: str, capability: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query only synced cassettes with specific capability.

        Args:
            query: Search query string
            capability: Required capability (e.g., 'vectors', 'ast', 'research')
            top_k: Results per cassette

        Returns:
            Dict mapping cassette_id to list of results
        """
        if self.verbose:
            print(f"[NETWORK] Routing query to cassettes with capability '{capability}': '{query}'", file=sys.stderr)

        results = {}
        for cassette_id, cassette in self.cassettes.items():
            if capability in cassette.capabilities:
                # Check sync status if enforcement is enabled
                if self.enforce_sync and not self._is_cassette_synced(cassette_id):
                    status = self._cassette_sync_status.get(cassette_id, BlanketStatus.UNSYNCED)
                    if self.verbose:
                        print(f"  {cassette_id}: SKIPPED (blanket {status.value})", file=sys.stderr)
                    results[cassette_id] = [{
                        'error': 'E_BLANKET_NOT_ALIGNED',
                        'blanket_status': status.value,
                        'message': 'Cassette sync required before query'
                    }]
                    continue

                try:
                    cassette_results = cassette.query(query, top_k)
                    results[cassette_id] = cassette_results
                    if self.verbose:
                        print(f"  {cassette_id}: {len(cassette_results)} results", file=sys.stderr)
                except Exception as e:
                    print(f"  {cassette_id}: ERROR - {e}", file=sys.stderr)
                    results[cassette_id] = []

        return results

    def get_network_status(self) -> Dict:
        """Get status of all registered cassettes including sync state."""
        # Count synced cassettes
        synced_count = sum(
            1 for status in self._cassette_sync_status.values()
            if status == BlanketStatus.ALIGNED
        )

        status = {
            "protocol_version": self.protocol_version,
            "cassette_count": len(self.cassettes),
            "synced_count": synced_count,
            "enforce_sync": self.enforce_sync,
            "hub_sync_tuple": self._hub_sync_tuple.to_dict() if self._hub_sync_tuple else None,
            "cassettes": {}
        }

        for cassette_id, cassette in self.cassettes.items():
            blanket_status = self._cassette_sync_status.get(cassette_id, BlanketStatus.UNSYNCED)
            status["cassettes"][cassette_id] = {
                "db_path": str(cassette.db_path),
                "db_exists": cassette.db_path.exists(),
                "capabilities": cassette.capabilities,
                "blanket_status": blanket_status.value,
                "stats": cassette.get_stats()
            }

        return status

    def get_sync_summary(self) -> Dict:
        """Get sync status summary for all cassettes.

        Returns:
            Dict with sync statistics and per-cassette status
        """
        aligned = []
        dissolved = []
        unsynced = []

        for cassette_id, status in self._cassette_sync_status.items():
            if status == BlanketStatus.ALIGNED:
                aligned.append(cassette_id)
            elif status == BlanketStatus.DISSOLVED:
                dissolved.append(cassette_id)
            else:
                unsynced.append(cassette_id)

        return {
            "hub_codebook_hash": self._hub_sync_tuple.codebook_sha256[:16] if self._hub_sync_tuple else None,
            "aligned": aligned,
            "dissolved": dissolved,
            "unsynced": unsynced,
            "total": len(self.cassettes),
            "aligned_count": len(aligned),
            "queryable": len(aligned) if self.enforce_sync else len(self.cassettes)
        }

    def resync_cassette(self, cassette_id: str) -> Dict:
        """Attempt to resync a dissolved or unsynced cassette.

        Args:
            cassette_id: Cassette to resync

        Returns:
            Dict with resync result
        """
        if cassette_id not in self.cassettes:
            return {
                'status': 'error',
                'error': 'E_CASSETTE_NOT_FOUND',
                'message': f'Cassette {cassette_id} not registered'
            }

        cassette = self.cassettes[cassette_id]
        handshake = cassette.handshake()
        sync_result = self._verify_cassette_sync(handshake)

        # Update status
        self._cassette_sync_status[cassette_id] = BlanketStatus(
            sync_result.get('blanket_status', 'UNSYNCED')
        )

        return {
            'status': 'resynced',
            'cassette_id': cassette_id,
            'sync_result': sync_result
        }

    def broadcast_resync(self) -> Dict:
        """Resync all cassettes in the network.

        Per CODEBOOK_SYNC_PROTOCOL Section 6.3: Network-wide sync.

        Returns:
            Dict with resync results for all cassettes
        """
        results = {}
        for cassette_id in list(self.cassettes.keys()):
            results[cassette_id] = self.resync_cassette(cassette_id)
        return results

    def print_network_status(self):
        """Print formatted network status including sync information."""
        sync_summary = self.get_sync_summary()

        print("\n" + "=" * 70)
        print("CASSETTE NETWORK STATUS")
        print("=" * 70)
        print(f"Protocol Version: {self.protocol_version}")
        print(f"Sync Enforcement: {self.enforce_sync}")
        print(f"Registered Cassettes: {len(self.cassettes)}")
        print(f"Aligned/Queryable: {sync_summary['aligned_count']}/{sync_summary['queryable']}")

        if self._hub_sync_tuple:
            print(f"\nHub Codebook: {self._hub_sync_tuple.codebook_id}")
            print(f"Hub Hash: {self._hub_sync_tuple.codebook_sha256[:16]}...")
            print(f"Hub Version: {self._hub_sync_tuple.codebook_semver}")

        print()
        for cassette_id, cassette in self.cassettes.items():
            stats = cassette.get_stats()
            blanket_status = self._cassette_sync_status.get(cassette_id, BlanketStatus.UNSYNCED)
            status_symbol = "✓" if blanket_status == BlanketStatus.ALIGNED else "✗"

            print(f"Cassette: {cassette_id} [{status_symbol} {blanket_status.value}]")
            print(f"  Database: {cassette.db_path.name}")
            print(f"  Exists: {cassette.db_path.exists()}")
            print(f"  Capabilities: {cassette.capabilities}")
            print(f"  Stats: {stats}")
            print()
