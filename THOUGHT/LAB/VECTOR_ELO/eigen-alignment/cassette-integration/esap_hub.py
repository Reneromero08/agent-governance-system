#!/usr/bin/env python3
"""
ESAP Network Hub - Cassette hub with spectral alignment verification.

Verifies spectral convergence during registration and groups cassettes
by alignment for cross-model queries.

Per Q35 (Markov Blankets): The handshake IS Active Inference:
1. PREDICTION: Sender predicts receiver has matching spectrum
2. VERIFICATION: Registration tests prediction
3. ERROR SIGNAL: Divergence = prediction error
4. ACTION: Assign to different alignment group or fail-closed
"""

import sys
from typing import Dict, List, Optional, Set

from network_hub import SemanticNetworkHub
from esap_cassette import ESAPCassetteMixin


class ESAPNetworkHub(SemanticNetworkHub):
    """Network hub with ESAP alignment verification.

    Groups cassettes by spectral similarity and verifies alignment
    before cross-cassette queries.
    """

    def __init__(self, verbose: bool = False, fail_closed: bool = True):
        """Initialize ESAP-enabled hub.

        Args:
            verbose: Enable verbose logging
            fail_closed: If True, reject queries to unaligned cassettes
        """
        super().__init__(verbose)
        self.alignment_groups: Dict[str, Set[str]] = {}  # group_id -> cassette_ids
        self.cassette_spectrums: Dict[str, Dict] = {}  # cassette_id -> spectrum
        self.fail_closed = fail_closed
        self.alignment_threshold = 0.95

    def register_cassette(self, cassette) -> Dict:
        """Register cassette with ESAP spectrum verification.

        Args:
            cassette: Cassette instance (may or may not support ESAP)

        Returns:
            Handshake metadata including ESAP info
        """
        # Get extended handshake with ESAP info
        if isinstance(cassette, ESAPCassetteMixin):
            handshake = cassette.esap_handshake()
        else:
            handshake = cassette.handshake()
            handshake["esap"] = {"enabled": False}

        cassette_id = handshake['cassette_id']

        # Store spectrum if available
        if handshake.get("esap", {}).get("enabled"):
            self.cassette_spectrums[cassette_id] = handshake["esap"]["spectrum"]
            self._update_alignment_groups(cassette_id)

        # Register in base hub
        self.cassettes[cassette_id] = cassette

        if self.verbose:
            print(f"[ESAP] Registered {cassette_id}", file=sys.stderr)
            if handshake.get("esap", {}).get("enabled"):
                print(f"  Spectrum: {handshake['esap']['spectrum']['anchor_hash']}", file=sys.stderr)
                print(f"  Effective rank: {handshake['esap']['spectrum']['effective_rank']:.2f}", file=sys.stderr)

        return handshake

    def _update_alignment_groups(self, new_cassette_id: str):
        """Assign cassette to alignment group based on spectral similarity.

        Args:
            new_cassette_id: ID of newly registered cassette
        """
        new_spectrum = self.cassette_spectrums.get(new_cassette_id)
        if not new_spectrum:
            return

        # Find matching group
        for group_id, members in self.alignment_groups.items():
            # Compare with first member of group
            first_member = next(iter(members))
            ref_spectrum = self.cassette_spectrums.get(first_member)
            if ref_spectrum:
                correlation = ESAPCassetteMixin.compute_spectrum_correlation(
                    new_spectrum, ref_spectrum
                )
                if correlation >= self.alignment_threshold:
                    self.alignment_groups[group_id].add(new_cassette_id)
                    if self.verbose:
                        print(f"  Aligned with group {group_id} (r={correlation:.3f})", file=sys.stderr)
                    return

        # Create new group
        group_id = f"group-{len(self.alignment_groups)}"
        self.alignment_groups[group_id] = {new_cassette_id}
        if self.verbose:
            print(f"  Created new alignment group: {group_id}", file=sys.stderr)

    def query_aligned(self, query: str, cassette_id: str, top_k: int = 10) -> Dict[str, List]:
        """Query only cassettes aligned with the reference cassette.

        This ensures semantic transfer is valid per Q35 Markov blanket semantics:
        only query across an aligned blanket boundary.

        Args:
            query: Search query
            cassette_id: Reference cassette to find alignment group
            top_k: Results per cassette

        Returns:
            Results from aligned cassettes only, or error if fail_closed
        """
        # Find alignment group for reference cassette
        aligned_ids = set()
        for group_id, members in self.alignment_groups.items():
            if cassette_id in members:
                aligned_ids = members
                break

        if not aligned_ids:
            if self.fail_closed:
                return {"error": "E_NO_ALIGNMENT_GROUP", "cassette_id": cassette_id}
            aligned_ids = {cassette_id}

        # Query aligned cassettes
        results = {}
        for cid in aligned_ids:
            if cid in self.cassettes:
                try:
                    results[cid] = self.cassettes[cid].query(query, top_k)
                except Exception as e:
                    results[cid] = {"error": str(e)}

        return results

    def verify_alignment(self, cassette_a: str, cassette_b: str) -> Dict:
        """Verify spectral alignment between two cassettes.

        Args:
            cassette_a: First cassette ID
            cassette_b: Second cassette ID

        Returns:
            Dict with aligned (bool), correlation, threshold, error info
        """
        spec_a = self.cassette_spectrums.get(cassette_a)
        spec_b = self.cassette_spectrums.get(cassette_b)

        if not spec_a or not spec_b:
            return {
                "aligned": False,
                "error": "E_SPECTRUM_NOT_FOUND",
                "missing": [c for c in [cassette_a, cassette_b]
                           if c not in self.cassette_spectrums]
            }

        correlation = ESAPCassetteMixin.compute_spectrum_correlation(spec_a, spec_b)

        return {
            "aligned": correlation >= self.alignment_threshold,
            "correlation": correlation,
            "threshold": self.alignment_threshold,
            "cassettes": [cassette_a, cassette_b]
        }

    def get_alignment_status(self) -> Dict:
        """Get status of all alignment groups.

        Returns:
            Dict with groups, unaligned cassettes, threshold, fail_closed mode
        """
        return {
            "groups": {
                gid: list(members)
                for gid, members in self.alignment_groups.items()
            },
            "unaligned": [
                cid for cid in self.cassettes
                if cid not in self.cassette_spectrums
            ],
            "threshold": self.alignment_threshold,
            "fail_closed": self.fail_closed,
            "total_cassettes": len(self.cassettes),
            "esap_enabled_count": len(self.cassette_spectrums)
        }

    def get_network_status(self) -> Dict:
        """Get enhanced network status including ESAP info.

        Returns:
            Dict with protocol version, cassettes, and ESAP alignment info
        """
        base_status = super().get_network_status()
        base_status["esap"] = self.get_alignment_status()
        return base_status
