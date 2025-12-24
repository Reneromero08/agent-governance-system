"""
CATALYTIC-DPT Primitives

Smallest possible catalytic computing kernel for PoC validation.

Phase 1: CATLAB (Catalytic Learning A-B Testing Lab)

Primitives:
- catalytic_store: Content-addressable storage (CAS)
- merkle: Merkle tree for root digests
- spectral_codec: Domain → spectrum encoding
- ledger: Append-only receipt storage
- validator: Schema validation + error vectors
- micro_orchestrator: Tiny model with weight updates

Usage:
    from CATALYTIC_DPT.PRIMITIVES import CatalyticStore, MerkleTree, Validator

Governance:
    - Determinism required (SHA-256, explicit seeds)
    - All operations logged to ledger
    - Restoration proofs enforced
    - No external dependencies (stdlib + json only)

Date: 2025-12-23
Status: Phase 1 Implementation
"""

__version__ = "1.0.0-phase1"
__all__ = [
    "CatalyticStore",
    "MerkleTree",
    "SpectralCodec",
    "Ledger",
    "Validator",
    "MicroOrchestrator",
]
