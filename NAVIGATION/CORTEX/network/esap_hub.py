#!/usr/bin/env python3
"""
ESAP-Enabled Network Hub - Spectral alignment verification for cassette networks.

Extends SemanticNetworkHub with ESAP handshake verification.
When cassettes with vector capabilities register, their spectral signatures
are checked for alignment according to the Spectral Convergence Theorem.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))
from network_hub import SemanticNetworkHub
from cassette_protocol import DatabaseCassette

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "THOUGHT/LAB/VECTOR_ELO/eigen-alignment"))
from lib.handshake import check_convergence, CONVERGENCE_THRESHOLD

import numpy as np


class ESAPNetworkHub(SemanticNetworkHub):
    """Network hub with ESAP spectral alignment verification.

    When cassettes with ESAP capability register, the hub:
    1. Extracts their spectrum signature
    2. Verifies spectral convergence with existing cassettes
    3. Groups aligned cassettes for cross-query optimization
    """

    def __init__(self, verbose: bool = False, require_alignment: bool = False):
        """Initialize ESAP-enabled hub.

        Args:
            verbose: Print debug info
            require_alignment: Reject cassettes that don't align
        """
        super().__init__(verbose=verbose)
        self.esap_version = "1.0.0"
        self.require_alignment = require_alignment
        self.spectra: Dict[str, dict] = {}  # cassette_id -> spectrum
        self.alignment_groups: Dict[str, List[str]] = {}  # group_id -> [cassette_ids]
        self.convergence_matrix: Dict[Tuple[str, str], float] = {}  # (a, b) -> correlation

    def register_cassette(self, cassette: DatabaseCassette) -> Dict:
        """Register cassette with ESAP verification.

        If the cassette supports ESAP (has esap_handshake method),
        verifies spectral alignment with existing cassettes.

        Args:
            cassette: Cassette to register

        Returns:
            Extended handshake metadata including ESAP info
        """
        # Get ESAP handshake if available
        if hasattr(cassette, 'esap_handshake'):
            handshake = cassette.esap_handshake()
        else:
            handshake = cassette.handshake()

        cassette_id = handshake['cassette_id']

        # Check for ESAP spectrum
        esap_info = handshake.get('esap', {})
        if esap_info.get('enabled'):
            spectrum = esap_info.get('spectrum', {})
            self.spectra[cassette_id] = spectrum

            # Verify alignment with existing cassettes
            alignment_results = self._verify_alignment_with_all(cassette_id, spectrum)
            handshake['esap']['alignment_results'] = alignment_results

            # Group aligned cassettes
            self._update_alignment_groups(cassette_id, alignment_results)

            if self.verbose:
                self._print_alignment_results(cassette_id, alignment_results)

            # Reject if required alignment not met
            if self.require_alignment and self.spectra:
                aligned_count = sum(1 for r in alignment_results.values() if r['converges'])
                if aligned_count == 0 and len(self.spectra) > 1:
                    print(f"[ESAP] REJECTED: {cassette_id} - no spectral convergence with existing cassettes")
                    return {'rejected': True, 'reason': 'SPECTRUM_DIVERGENCE', **handshake}

        # Register in parent
        self.cassettes[cassette_id] = cassette

        if self.verbose:
            print(f"[ESAP] Registered: {cassette_id}")
            if esap_info.get('enabled'):
                df = esap_info['spectrum'].get('effective_rank', 0)
                print(f"  Effective rank (Df): {df:.2f}")

        return handshake

    def _verify_alignment_with_all(self, new_id: str, new_spectrum: dict) -> Dict[str, dict]:
        """Verify spectral convergence with all existing ESAP cassettes.

        Args:
            new_id: ID of new cassette
            new_spectrum: Spectrum from new cassette

        Returns:
            Dict mapping existing cassette IDs to convergence results
        """
        results = {}

        new_cv = np.array(new_spectrum.get('cumulative_variance', []))
        new_df = new_spectrum.get('effective_rank', 0)

        for existing_id, existing_spectrum in self.spectra.items():
            if existing_id == new_id:
                continue

            existing_cv = np.array(existing_spectrum.get('cumulative_variance', []))
            existing_df = existing_spectrum.get('effective_rank', 0)

            if len(new_cv) == 0 or len(existing_cv) == 0:
                results[existing_id] = {'converges': False, 'reason': 'missing_spectrum'}
                continue

            convergence = check_convergence(new_cv, existing_cv, new_df, existing_df)
            results[existing_id] = convergence

            # Store in convergence matrix (symmetric)
            key = tuple(sorted([new_id, existing_id]))
            self.convergence_matrix[key] = convergence['correlation']

        return results

    def _update_alignment_groups(self, cassette_id: str, alignment_results: Dict[str, dict]):
        """Update alignment groups based on convergence results.

        Cassettes with correlation > threshold are grouped together.
        """
        # Find existing groups this cassette aligns with
        aligned_groups = set()

        for other_id, result in alignment_results.items():
            if result.get('converges'):
                # Find which group the other cassette is in
                for group_id, members in self.alignment_groups.items():
                    if other_id in members:
                        aligned_groups.add(group_id)
                        break

        if aligned_groups:
            # Join existing group(s)
            if len(aligned_groups) == 1:
                # Add to existing group
                group_id = aligned_groups.pop()
                self.alignment_groups[group_id].append(cassette_id)
            else:
                # Merge groups (rare case)
                merged = []
                for group_id in aligned_groups:
                    merged.extend(self.alignment_groups.pop(group_id))
                merged.append(cassette_id)
                new_group_id = f"group_{len(self.alignment_groups)}"
                self.alignment_groups[new_group_id] = merged
        else:
            # Create new group
            group_id = f"group_{len(self.alignment_groups)}"
            self.alignment_groups[group_id] = [cassette_id]

    def _print_alignment_results(self, cassette_id: str, results: Dict[str, dict]):
        """Print alignment verification results."""
        if not results:
            return

        print(f"[ESAP] Alignment check for {cassette_id}:")
        for other_id, result in results.items():
            r = result.get('correlation', 0)
            converges = result.get('converges', False)
            status = "ALIGNED" if converges else "DIVERGENT"
            print(f"  {cassette_id} <-> {other_id}: r={r:.4f} [{status}]")

    def query_aligned(self, query: str, cassette_id: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query only cassettes aligned with the specified one.

        Args:
            query: Search query
            cassette_id: Reference cassette ID
            top_k: Results per cassette

        Returns:
            Results from aligned cassettes only
        """
        # Find the alignment group
        for group_id, members in self.alignment_groups.items():
            if cassette_id in members:
                return self._query_group(query, members, top_k)

        # Fallback: query just this cassette
        if cassette_id in self.cassettes:
            results = self.cassettes[cassette_id].query(query, top_k)
            return {cassette_id: results}

        return {}

    def _query_group(self, query: str, members: List[str], top_k: int) -> Dict[str, List[dict]]:
        """Query all cassettes in an alignment group."""
        results = {}
        for member_id in members:
            if member_id in self.cassettes:
                try:
                    results[member_id] = self.cassettes[member_id].query(query, top_k)
                except Exception as e:
                    print(f"[ESAP] Query error for {member_id}: {e}")
                    results[member_id] = []
        return results

    def get_alignment_matrix(self) -> Dict:
        """Get full alignment matrix for visualization."""
        cassette_ids = list(self.spectra.keys())
        n = len(cassette_ids)

        matrix = {}
        for i, id_a in enumerate(cassette_ids):
            matrix[id_a] = {}
            for j, id_b in enumerate(cassette_ids):
                if i == j:
                    matrix[id_a][id_b] = 1.0
                else:
                    key = tuple(sorted([id_a, id_b]))
                    matrix[id_a][id_b] = self.convergence_matrix.get(key, 0.0)

        return {
            "cassettes": cassette_ids,
            "matrix": matrix,
            "groups": self.alignment_groups,
            "threshold": CONVERGENCE_THRESHOLD
        }

    def print_esap_status(self):
        """Print ESAP alignment status."""
        print("\n" + "=" * 70)
        print("ESAP ALIGNMENT STATUS")
        print("=" * 70)
        print(f"ESAP Version: {self.esap_version}")
        print(f"Convergence Threshold: {CONVERGENCE_THRESHOLD}")
        print(f"ESAP-Enabled Cassettes: {len(self.spectra)}")
        print(f"Alignment Groups: {len(self.alignment_groups)}")

        for group_id, members in self.alignment_groups.items():
            print(f"\n{group_id}:")
            for member in members:
                df = self.spectra.get(member, {}).get('effective_rank', 0)
                print(f"  - {member} (Df={df:.2f})")

        if self.convergence_matrix:
            print("\nConvergence Matrix:")
            for (a, b), r in sorted(self.convergence_matrix.items()):
                status = "ALIGNED" if r >= CONVERGENCE_THRESHOLD else "DIVERGENT"
                print(f"  {a} <-> {b}: r={r:.4f} [{status}]")

        print()
