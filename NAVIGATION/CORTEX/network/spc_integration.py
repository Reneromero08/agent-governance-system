#!/usr/bin/env python3
"""
SPC Integration - Bridges SPC Decoder with Memory Cassette.

This module wires together:
- spc_decoder: Pointer resolution (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR)
- memory_cassette: CAS storage for HASH_PTR content
- codebook_sync: Sync protocol (Phase 4.2)
- spc_metrics: CDR/ECR tracking (Q33)

Phase 4.2 Complete Integration per:
- LAW/CANON/SEMANTIC/SPC_SPEC.md (Pointer Types)
- LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md (Sync Protocol)
- Q35 (Markov Blankets): Blanket alignment gating
- Q33 (Semantic Density): CDR measurement
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# Handle both package and direct imports
try:
    from .spc_decoder import (
        SPCDecoder,
        pointer_resolve,
        register_cas_lookup,
        unregister_cas_lookup,
        is_cas_available,
        DecodeSuccess,
        FailClosed,
        PointerType,
        ErrorCode
    )
    from .memory_cassette import MemoryCassette
    from .spc_metrics import SPCMetricsTracker, get_metrics_tracker
    from .codebook_sync import CodebookSync, SyncTuple, BlanketStatus
except ImportError:
    from spc_decoder import (
        SPCDecoder,
        pointer_resolve,
        register_cas_lookup,
        unregister_cas_lookup,
        is_cas_available,
        DecodeSuccess,
        FailClosed,
        PointerType,
        ErrorCode
    )
    from memory_cassette import MemoryCassette
    from spc_metrics import SPCMetricsTracker, get_metrics_tracker
    from codebook_sync import CodebookSync, SyncTuple, BlanketStatus

# Project root for default paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SPCIntegration:
    """Integrated SPC system with CAS storage.

    Combines:
    - SPC Decoder for pointer resolution
    - Memory Cassette for content-addressed storage
    - Pointer caching for performance
    - Q33/Q35 metrics integration
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        codebook_path: Optional[Path] = None,
        auto_register: bool = True,
        track_metrics: bool = True
    ):
        """Initialize integrated SPC system.

        Args:
            db_path: Path to memory cassette database
            codebook_path: Path to CODEBOOK.json
            auto_register: If True, automatically register CAS lookup
            track_metrics: If True, track CDR/ECR metrics (Q33)
        """
        self.db_path = db_path or PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "memory.db"
        self.codebook_path = codebook_path or PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"

        # Initialize components
        self.decoder = SPCDecoder(self.codebook_path)
        self.memory = MemoryCassette(self.db_path)

        # Track integration state
        self._cas_registered = False
        self._blanket_status = "UNSYNCED"
        self._sync_tuple = None

        # Phase 4.2: Codebook sync protocol
        self._sync_protocol = CodebookSync(sender_id="spc-integration")

        # Phase 4.2: CDR/ECR metrics tracking (Q33)
        self._track_metrics = track_metrics
        self._metrics_tracker = SPCMetricsTracker() if track_metrics else None

        if auto_register:
            self.enable_cas()

    def enable_cas(self) -> None:
        """Enable CAS lookup for HASH_PTR resolution."""
        if not self._cas_registered:
            register_cas_lookup(self.memory.cas_lookup)
            self._cas_registered = True

    def disable_cas(self) -> None:
        """Disable CAS lookup."""
        if self._cas_registered:
            unregister_cas_lookup()
            self._cas_registered = False

    def sync_handshake(self) -> Dict:
        """Perform sync handshake per CODEBOOK_SYNC_PROTOCOL.

        Establishes Markov blanket alignment (Q35).
        Updates metrics tracker blanket status.

        Returns:
            Dict with sync result including blanket_status and metrics
        """
        # Build decoder's sync tuple
        decoder_tuple = SyncTuple(
            codebook_id="ags-codebook",
            codebook_sha256=self.decoder.codebook_hash,
            codebook_semver=self.decoder.codebook.get("version", "0.0.0"),
            kernel_version=self.decoder.KERNEL_VERSION,
            tokenizer_id=self.decoder.TOKENIZER_ID
        )

        # Get memory cassette's sync tuple (via cassette protocol)
        cassette_tuple_data = self.memory.get_sync_tuple() if hasattr(self.memory, 'get_sync_tuple') else decoder_tuple.to_dict()
        cassette_tuple = SyncTuple.from_dict(cassette_tuple_data)

        # Create sync request via protocol
        sync_request = self._sync_protocol.create_sync_request(
            decoder_tuple,
            capabilities=["symbol_ptr", "hash_ptr", "composite_ptr"]
        )

        # Create response (cassette as receiver)
        sync_response = self._sync_protocol.create_sync_response(
            sync_request,
            cassette_tuple
        )

        # Verify response
        is_valid, message = self._sync_protocol.verify_sync_response(sync_request, sync_response)

        if is_valid:
            self._blanket_status = "ALIGNED"
            self._sync_tuple = decoder_tuple.to_dict()
            # Update metrics tracker blanket status
            if self._metrics_tracker:
                self._metrics_tracker.set_blanket_status("ALIGNED")
        else:
            self._blanket_status = "DISSOLVED"
            self._sync_tuple = decoder_tuple.to_dict()
            # Invalidate pointer cache on mismatch
            self.memory.pointer_invalidate(codebook_id=decoder_tuple.codebook_id)
            # Update metrics tracker blanket status
            if self._metrics_tracker:
                self._metrics_tracker.set_blanket_status("DISSOLVED")

        # Compute R-value for health tracking
        r_value = self._sync_protocol.compute_continuous_r(decoder_tuple, cassette_tuple)

        return {
            "status": "MATCHED" if self._blanket_status == "ALIGNED" else "MISMATCHED",
            "blanket_status": self._blanket_status,
            "sync_tuple": decoder_tuple.to_dict(),
            "r_value": round(r_value, 4),
            "message": message,
            "session_token": sync_response.get("session_token")
        }

    def resolve(
        self,
        pointer: str,
        context_keys: Optional[Dict] = None,
        cache: bool = True,
        record_metrics: bool = True
    ) -> Dict:
        """Resolve a pointer with full integration.

        Args:
            pointer: SPC pointer (e.g., "C3", "sha256:abc...", "法.驗")
            context_keys: Context keys for disambiguation
            cache: Whether to cache the result
            record_metrics: Whether to record CDR/ECR metrics

        Returns:
            Dict with resolution result
        """
        # Check blanket alignment (Q35)
        if self._blanket_status != "ALIGNED":
            self.sync_handshake()

        # Check cache first
        if cache:
            cached = self.memory.pointer_lookup(pointer)
            if cached:
                return {
                    "status": "SUCCESS",
                    "source": "cache",
                    "cached": cached
                }

        # Resolve pointer
        result = pointer_resolve(
            pointer,
            context_keys=context_keys or {},
            codebook_sha256=self._sync_tuple["codebook_sha256"] if self._sync_tuple else None
        )

        # Cache successful resolutions
        if result.get("status") == "SUCCESS" and cache:
            # Determine pointer type
            if pointer.startswith("sha256:"):
                ptr_type = "hash"
            elif len(pointer) == 1:
                ptr_type = "symbol"
            else:
                ptr_type = "composite"

            # Compute content hash for caching
            expansion = result.get("ir", {}).get("inputs", {}).get("expansion", {})
            content_str = json.dumps(expansion, sort_keys=True)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

            self.memory.pointer_register(
                pointer=pointer,
                pointer_type=ptr_type,
                target_hash=content_hash,
                codebook_id=self._sync_tuple["codebook_id"] if self._sync_tuple else "ags-codebook"
            )

            # Record metrics (Q33 CDR/ECR tracking)
            if record_metrics and self._metrics_tracker and self._track_metrics:
                expansion_text = self._get_expansion_text(expansion)
                ir_node = result.get("ir", {}).get("inputs", {}).get("expansion")
                self._metrics_tracker.record(
                    pointer=pointer,
                    expansion=expansion_text,
                    ir_node=ir_node,
                    correct=True  # Successful resolution is correct
                )

        # Record failed resolutions as incorrect (for ECR)
        elif result.get("status") == "FAIL_CLOSED" and self._metrics_tracker and self._track_metrics:
            self._metrics_tracker.global_metrics.total_expansions += 1
            # Don't increment correct_expansions

        return result

    def _get_expansion_text(self, expansion: Dict) -> str:
        """Extract text from expansion for metrics measurement.

        Args:
            expansion: Expansion dictionary from resolution

        Returns:
            Text representation of expansion
        """
        if isinstance(expansion, str):
            return expansion

        # Try common fields
        if 'full' in expansion:
            return expansion['full']
        if 'summary' in expansion:
            return expansion['summary']
        if 'text' in expansion:
            return expansion['text']

        # Fallback to JSON
        return json.dumps(expansion, sort_keys=True)

    def store_content(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        context: str = "spc_content"
    ) -> Tuple[str, str]:
        """Store content and return hash for HASH_PTR.

        Args:
            text: Content to store
            metadata: Optional metadata
            context: Memory context

        Returns:
            Tuple of (hash, pointer) where pointer is "sha256:<hash>"
        """
        result = self.memory.memory_store(
            text=text,
            context=context,
            metadata=metadata
        )
        hash_value = result["memory_hash"]
        pointer = f"sha256:{hash_value}"
        return hash_value, pointer

    def get_stats(self) -> Dict:
        """Get integration statistics including CDR/ECR metrics.

        Returns:
            Dict with decoder, memory, pointer cache, and metrics stats
        """
        stats = {
            "blanket_status": self._blanket_status,
            "sync_tuple": self._sync_tuple,
            "cas_enabled": self._cas_registered,
            "pointer_cache": self.memory.pointer_stats(),
            "memory_stats": self.memory.get_stats()
        }

        # Add CDR/ECR metrics (Q33)
        if self._metrics_tracker:
            stats["metrics"] = self._metrics_tracker.get_report()

        return stats

    def get_metrics_report(self) -> Dict:
        """Get detailed CDR/ECR metrics report (Q33).

        Returns:
            Dict with comprehensive metrics including per-symbol breakdown
        """
        if not self._metrics_tracker:
            return {"error": "Metrics tracking disabled"}

        return self._metrics_tracker.get_report()

    def reset_metrics(self) -> None:
        """Reset CDR/ECR metrics tracker."""
        if self._metrics_tracker:
            self._metrics_tracker = SPCMetricsTracker()
            if self._blanket_status == "ALIGNED":
                self._metrics_tracker.set_blanket_status("ALIGNED")


# ============================================================================
# Convenience Functions
# ============================================================================

_integration: Optional[SPCIntegration] = None


def get_spc_integration(
    db_path: Optional[Path] = None,
    auto_init: bool = True
) -> SPCIntegration:
    """Get or create global SPC integration instance.

    Args:
        db_path: Optional custom database path
        auto_init: Whether to auto-initialize if not exists

    Returns:
        SPCIntegration instance
    """
    global _integration

    if _integration is None and auto_init:
        _integration = SPCIntegration(db_path=db_path)

    return _integration


def resolve_pointer(
    pointer: str,
    context_keys: Optional[Dict] = None
) -> Dict:
    """Convenience function to resolve a pointer.

    Args:
        pointer: SPC pointer
        context_keys: Optional context keys

    Returns:
        Resolution result
    """
    integration = get_spc_integration()
    return integration.resolve(pointer, context_keys)


def store_for_hash_ptr(
    text: str,
    metadata: Optional[Dict] = None
) -> str:
    """Store content and return a HASH_PTR.

    Args:
        text: Content to store
        metadata: Optional metadata

    Returns:
        HASH_PTR string (sha256:...)
    """
    integration = get_spc_integration()
    _, pointer = integration.store_content(text, metadata)
    return pointer


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SPCIntegration",
    "get_spc_integration",
    "resolve_pointer",
    "store_for_hash_ptr",
]
