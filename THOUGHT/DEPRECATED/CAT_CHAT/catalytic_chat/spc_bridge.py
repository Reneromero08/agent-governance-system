"""
SPC Bridge - Connects Semantic Pointer Compression to CAT Chat.

Bridges the main repo's SPCDecoder to CAT Chat's resolution chain.

Provides:
- Codebook sync handshake for session start (D.1)
- SPC pointer detection and resolution (D.2)
- Compression metrics tracking (D.3)

Per CODEBOOK_SYNC_PROTOCOL.md:
- Sync tuple: (codebook_id, codebook_sha256, kernel_version)
- States: UNSYNCED -> ALIGNED or DISSOLVED
- Fail-closed on mismatch
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add NAVIGATION/CORTEX/network to path for SPCDecoder import
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SPC_PATH = _PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "network"
if str(_SPC_PATH) not in sys.path:
    sys.path.insert(0, str(_SPC_PATH))

# Import SPC components
from spc_decoder import (
    SPCDecoder,
    PointerType,
    ErrorCode,
    DecodeSuccess,
    FailClosed,
    pointer_resolve,
)


@dataclass
class SPCCompressionMetrics:
    """
    Track SPC compression metrics per session.

    Metrics per Q33 (Semantic Density):
    - CDR = Concept Density Ratio = concept_units / pointer_tokens
    - Compression ratio = tokens_expanded / tokens_pointers
    """

    total_resolutions: int = 0
    spc_resolutions: int = 0
    tokens_expanded: int = 0  # Total tokens of expanded content
    tokens_pointers: int = 0  # Total tokens of pointers
    tokens_saved: int = 0  # Difference
    cdr_samples: List[float] = field(default_factory=list)
    symbol_usage: Dict[str, int] = field(default_factory=dict)
    symbol_savings: Dict[str, int] = field(default_factory=dict)

    def record_resolution(
        self,
        pointer: str,
        expansion_text: str,
        pointer_tokens: int,
        expansion_tokens: int,
        concept_units: int = 2,
    ) -> None:
        """Record a single SPC resolution."""
        self.total_resolutions += 1
        self.spc_resolutions += 1

        self.tokens_expanded += expansion_tokens
        self.tokens_pointers += pointer_tokens
        self.tokens_saved += expansion_tokens - pointer_tokens

        # CDR = concept_units / pointer_tokens
        if pointer_tokens > 0:
            cdr = concept_units / pointer_tokens
            self.cdr_samples.append(cdr)

        # Track per-symbol stats
        base_symbol = pointer[0] if pointer else "?"
        self.symbol_usage[base_symbol] = self.symbol_usage.get(base_symbol, 0) + 1
        self.symbol_savings[base_symbol] = (
            self.symbol_savings.get(base_symbol, 0) + expansion_tokens - pointer_tokens
        )

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio (expanded/pointer)."""
        if self.tokens_pointers > 0:
            return self.tokens_expanded / self.tokens_pointers
        return 0.0

    @property
    def average_cdr(self) -> float:
        """Average Concept Density Ratio."""
        if self.cdr_samples:
            return sum(self.cdr_samples) / len(self.cdr_samples)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event logging."""
        return {
            "total_resolutions": self.total_resolutions,
            "spc_resolutions": self.spc_resolutions,
            "tokens_expanded": self.tokens_expanded,
            "tokens_pointers": self.tokens_pointers,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round(self.compression_ratio, 2),
            "average_cdr": round(self.average_cdr, 2),
            "symbol_usage": self.symbol_usage,
            "symbol_savings": self.symbol_savings,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class SPCBridge:
    """
    Bridge SPC system to CAT Chat resolution chain.

    Implements D.1-D.3 from CAT_CHAT_ROADMAP_2.0.md:
    - D.1: Codebook sync handshake with fail-closed semantics
    - D.2: Pointer resolution (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR)
    - D.3: Compression metrics tracking
    """

    # Sync states per CODEBOOK_SYNC_PROTOCOL.md
    STATE_UNSYNCED = "UNSYNCED"
    STATE_ALIGNED = "ALIGNED"
    STATE_DISSOLVED = "DISSOLVED"

    # ASCII radicals from SPC_SPEC
    ASCII_RADICALS = set("CIVLGSRAJP")

    # CJK glyphs from SPCDecoder
    CJK_GLYPHS = set(SPCDecoder.CJK_GLYPHS.keys())

    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize SPC bridge.

        Args:
            repo_root: Repository root path (default: auto-detect)
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]
        self.repo_root = repo_root

        # Lazy-loaded decoder
        self._decoder: Optional[SPCDecoder] = None

        # Sync state
        self._sync_status: str = self.STATE_UNSYNCED
        self._sync_tuple: Optional[Dict[str, str]] = None
        self._session_id: Optional[str] = None

        # Metrics
        self._metrics = SPCCompressionMetrics()

    @property
    def decoder(self) -> SPCDecoder:
        """Lazy load SPCDecoder."""
        if self._decoder is None:
            self._decoder = SPCDecoder()
        return self._decoder

    @property
    def is_aligned(self) -> bool:
        """Check if codebook is aligned."""
        return self._sync_status == self.STATE_ALIGNED

    def sync_handshake(self, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform codebook sync handshake per CODEBOOK_SYNC_PROTOCOL.md.

        This verifies that the codebook is loaded and computes its hash.
        The session is bound to this codebook state.

        Args:
            session_id: Session ID to bind

        Returns:
            Tuple of (success, sync_result_dict)
            - success: True if ALIGNED, False if DISSOLVED
            - sync_result_dict contains: codebook_id, codebook_sha256, status, etc.
        """
        now = datetime.now(timezone.utc).isoformat()

        try:
            # Load codebook and compute hash
            codebook_hash = self.decoder.codebook_hash
            codebook_version = self.decoder.codebook.get("version", "unknown")
            kernel_version = SPCDecoder.KERNEL_VERSION

            # Build sync tuple
            sync_tuple = {
                "codebook_id": "ags-codebook",
                "codebook_sha256": codebook_hash,
                "codebook_version": codebook_version,
                "kernel_version": kernel_version,
                "tokenizer_id": SPCDecoder.TOKENIZER_ID,
            }

            # Mark as aligned
            self._sync_status = self.STATE_ALIGNED
            self._sync_tuple = sync_tuple
            self._session_id = session_id

            return True, {
                "status": "MATCHED",
                "sync_tuple": sync_tuple,
                "blanket_status": self.STATE_ALIGNED,
                "timestamp": now,
                "message": "Codebook sync successful",
            }

        except Exception as e:
            # Fail-closed: any error during sync = DISSOLVED
            self._sync_status = self.STATE_DISSOLVED

            return False, {
                "status": "FAILED",
                "blanket_status": self.STATE_DISSOLVED,
                "timestamp": now,
                "error": str(e),
                "message": f"Codebook sync failed: {e}",
            }

    def is_spc_pointer(self, pointer: str) -> bool:
        """
        Check if a string looks like an SPC pointer.

        Detects:
        - HASH_PTR: sha256:abc123...
        - SYMBOL_PTR: Single CJK glyph (法, 真, etc.) or ASCII radical (C, I, V)
        - COMPOSITE_PTR: C3, C&I, 法.驗, L.C.3, C:build

        Args:
            pointer: String to check

        Returns:
            True if this looks like an SPC pointer
        """
        if not pointer or not isinstance(pointer, str):
            return False

        pointer = pointer.strip()
        if not pointer:
            return False

        # HASH_PTR: sha256:...
        if pointer.startswith("sha256:"):
            return True

        # SYMBOL_PTR: Single CJK glyph
        if len(pointer) == 1 and pointer in self.CJK_GLYPHS:
            return True

        # SYMBOL_PTR: Single ASCII radical
        if len(pointer) == 1 and pointer in self.ASCII_RADICALS:
            return True

        # COMPOSITE_PTR: Starts with ASCII radical + something
        if pointer and pointer[0] in self.ASCII_RADICALS and len(pointer) > 1:
            # C3, C*, C!, C?, C&I, C|I, C:build, etc.
            return True

        # COMPOSITE_PTR: CJK path like 法.驗
        if any(c in pointer for c in self.CJK_GLYPHS) and "." in pointer:
            return True

        # COMPOSITE_PTR: ASCII path like L.C.3
        if pointer[0] in self.ASCII_RADICALS and "." in pointer:
            return True

        return False

    def resolve_pointer(
        self, pointer: str, context_keys: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve an SPC pointer via SPCDecoder.

        Args:
            pointer: The SPC pointer to resolve
            context_keys: Optional context for disambiguation

        Returns:
            Resolution result dict or None if resolution fails
        """
        if not self.is_aligned:
            return None

        if not self.is_spc_pointer(pointer):
            return None

        # Use the decoder with codebook hash verification
        result = self.decoder.decode(
            pointer,
            context_keys=context_keys or {},
            codebook_sha256=self._sync_tuple["codebook_sha256"]
            if self._sync_tuple
            else None,
        )

        if isinstance(result, DecodeSuccess):
            # Extract expansion for metrics
            expansion_text = self._extract_expansion_text(result.ir)
            pointer_tokens = result.token_receipt.get("tokens_in", len(pointer))
            expansion_tokens = result.token_receipt.get(
                "tokens_out", len(expansion_text.split())
            )
            concept_units = result.token_receipt.get("concept_units", 2)

            # Record metrics
            self._metrics.record_resolution(
                pointer=pointer,
                expansion_text=expansion_text,
                pointer_tokens=pointer_tokens,
                expansion_tokens=expansion_tokens,
                concept_units=concept_units,
            )

            return {
                "status": "SUCCESS",
                "ir": result.ir,
                "token_receipt": result.token_receipt,
                "expansion": expansion_text,
            }

        # FailClosed result
        return None

    def _extract_expansion_text(self, ir: Dict[str, Any]) -> str:
        """Extract text content from SPC IR."""
        expansion = ir.get("inputs", {}).get("expansion", {})

        if isinstance(expansion, str):
            return expansion

        # Try various fields in order of preference
        for field in ["full", "summary", "text", "content"]:
            if field in expansion:
                val = expansion[field]
                if isinstance(val, str):
                    return val

        # For domain/path expansions, return the path
        if "path" in expansion:
            return str(expansion["path"])

        # For rule sets, return the rules list
        if "rules" in expansion:
            return ", ".join(expansion["rules"])

        # Fallback: serialize the expansion
        import json

        return json.dumps(expansion, sort_keys=True)

    def get_expansion_text(self, result: Dict[str, Any]) -> str:
        """
        Extract text content from resolution result.

        Args:
            result: Result from resolve_pointer()

        Returns:
            Expansion text string
        """
        if "expansion" in result:
            return result["expansion"]

        if "ir" in result:
            return self._extract_expansion_text(result["ir"])

        return ""

    def get_metrics(self) -> Dict[str, Any]:
        """Get current compression metrics."""
        return self._metrics.to_dict()

    def get_sync_tuple(self) -> Optional[Dict[str, str]]:
        """Get the current sync tuple."""
        return self._sync_tuple

    def reset_metrics(self) -> None:
        """Reset compression metrics."""
        self._metrics = SPCCompressionMetrics()


# Module-level convenience function
def create_spc_bridge(repo_root: Optional[Path] = None) -> SPCBridge:
    """Create a new SPC bridge instance."""
    return SPCBridge(repo_root)


__all__ = [
    "SPCBridge",
    "SPCCompressionMetrics",
    "create_spc_bridge",
]
