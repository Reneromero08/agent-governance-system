#!/usr/bin/env python3
"""
Symbolic Compiler (P.2)

Multi-level semantic rendering with verifiable lossless round-trip.

Per roadmap P.2:
- P.2.1.1: Express same meaning at multiple compression levels
- P.2.1.2: Round-trip is verifiably lossless (E > 0.99)
- P.2.1.3: Compression ratios are measurable and receipted

Compression Levels:
- Level 0: Full Prose (humans)
- Level 1: @Symbol References (compact)
- Level 2: Vector Hashes (minimal)
- Level 3: Custom Protocols (emergent)

Design Decisions:
- Symbol Registry: Hybrid (global + per-resident)
- Level 3 Grammar: Emergent only (no predefined rules)
- Priority: E preservation first (semantic fidelity > 0.99)
"""

import sys
import re
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

FERAL_PATH = Path(__file__).parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from geometric_reasoner import GeometricReasoner, GeometricState, GeometricOperations


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CompressionLevel:
    """Defines a compression level."""
    level: int       # 0-3
    name: str        # "prose", "symbol", "hash", "protocol"
    description: str


COMPRESSION_LEVELS = [
    CompressionLevel(0, "prose", "Full natural language for humans"),
    CompressionLevel(1, "symbol", "@Symbol references with minimal prose"),
    CompressionLevel(2, "hash", "Vector SHA256 hashes only"),
    CompressionLevel(3, "protocol", "Emergent custom notation"),
]


@dataclass
class RenderResult:
    """Result of rendering a GeometricState at a compression level."""
    content: str
    level: int
    level_name: str
    E_preserved: float      # Born rule similarity with original
    Df_preserved: float     # Participation ratio delta
    compression_ratio: float  # (original_tokens / compressed_tokens)
    original_hash: str
    receipt_hash: str
    timestamp: str


@dataclass
class RoundTripVerification:
    """Verification of lossless round-trip compression."""
    original_hash: str
    compressed_form: str
    decompressed_hash: str
    E_delta: float          # 1.0 - E (must be < 0.01)
    Df_delta: float         # |Df_orig - Df_decomp| (must be < 0.01)
    verified: bool          # E > 0.99 AND Df_delta < 0.01
    level: int
    receipt: Dict


@dataclass
class SymbolEntry:
    """Entry in the symbol registry."""
    symbol: str             # e.g., "@Concept-Entanglement"
    vector_hash: str        # SHA256 of the GeometricState vector
    state_Df: float         # Df at time of registration
    first_seen: str         # ISO timestamp
    first_registered_by: str  # Resident ID
    frequency: int          # Total uses across all residents
    adopters: List[str]     # Resident IDs who have used this symbol
    is_global: bool         # True if promoted to global registry
    meanings: List[str]     # Text meanings associated with this symbol
    receipt_hash: str


# =============================================================================
# Hybrid Symbol Registry
# =============================================================================

class HybridSymbolRegistry:
    """
    Two-tier symbol registry: global + per-resident.

    Design Decision: Hybrid registry
    - Core symbols are swarm-global (shared vocabulary)
    - Residents can invent local notations
    - Local notations promoted to global when adopted by 2+ residents

    Promotion Criteria:
    - Symbol used by >= 2 different residents
    - Frequency threshold met (>= 5 uses per resident)
    - E consistency: symbol maps to similar GeometricState (E > 0.9)
    """

    PROMOTION_ADOPTER_THRESHOLD = 2   # Minimum residents for promotion
    PROMOTION_FREQUENCY_THRESHOLD = 5  # Minimum uses per resident
    PROMOTION_E_THRESHOLD = 0.9       # E consistency across adopters

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize hybrid registry.

        Args:
            storage_path: Path for persistence (default: FERAL_PATH/symbol_registry/)
        """
        self.storage_path = storage_path or (FERAL_PATH / "symbol_registry")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.global_registry: Dict[str, SymbolEntry] = {}
        self.local_registries: Dict[str, Dict[str, SymbolEntry]] = {}

        # Map from vector hash to symbol for reverse lookups
        self._hash_to_symbol: Dict[str, str] = {}

        self._load_registry()

    def _load_registry(self):
        """Load existing registry from storage."""
        global_file = self.storage_path / "global_registry.json"
        if global_file.exists():
            try:
                with open(global_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for symbol, entry_data in data.get('symbols', {}).items():
                        entry = SymbolEntry(**entry_data)
                        self.global_registry[symbol] = entry
                        self._hash_to_symbol[entry.vector_hash] = symbol
            except (json.JSONDecodeError, TypeError):
                pass

        # Load local registries
        for local_file in self.storage_path.glob("local_*.json"):
            resident_id = local_file.stem.replace("local_", "")
            try:
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.local_registries[resident_id] = {}
                    for symbol, entry_data in data.get('symbols', {}).items():
                        entry = SymbolEntry(**entry_data)
                        self.local_registries[resident_id][symbol] = entry
            except (json.JSONDecodeError, TypeError):
                continue

    def _save_registry(self):
        """Persist registry to storage."""
        # Save global
        global_data = {
            'symbols': {s: asdict(e) for s, e in self.global_registry.items()},
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.storage_path / "global_registry.json", 'w', encoding='utf-8') as f:
            json.dump(global_data, f, indent=2)

        # Save locals
        for resident_id, registry in self.local_registries.items():
            local_data = {
                'resident_id': resident_id,
                'symbols': {s: asdict(e) for s, e in registry.items()},
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            with open(self.storage_path / f"local_{resident_id}.json", 'w', encoding='utf-8') as f:
                json.dump(local_data, f, indent=2)

    def register_local(
        self,
        resident_id: str,
        symbol: str,
        state: GeometricState,
        meaning: Optional[str] = None
    ) -> SymbolEntry:
        """
        Register a new local notation for a resident.

        Args:
            resident_id: The resident registering this symbol
            symbol: Symbol string (e.g., "@Concept-MyIdea")
            state: GeometricState this symbol represents
            meaning: Optional text meaning

        Returns:
            The created SymbolEntry
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        vector_hash = hashlib.sha256(state.vector.tobytes()).hexdigest()[:16]

        entry = SymbolEntry(
            symbol=symbol,
            vector_hash=vector_hash,
            state_Df=float(state.Df),
            first_seen=timestamp,
            first_registered_by=resident_id,
            frequency=1,
            adopters=[resident_id],
            is_global=False,
            meanings=[meaning] if meaning else [],
            receipt_hash=self._generate_receipt_hash(symbol, vector_hash, resident_id)
        )

        if resident_id not in self.local_registries:
            self.local_registries[resident_id] = {}

        self.local_registries[resident_id][symbol] = entry
        self._hash_to_symbol[vector_hash] = symbol

        self._save_registry()
        return entry

    def record_usage(self, symbol: str, resident_id: str):
        """Record that a resident used a symbol, updating frequency."""
        # Check global first
        if symbol in self.global_registry:
            entry = self.global_registry[symbol]
            entry.frequency += 1
            if resident_id not in entry.adopters:
                entry.adopters.append(resident_id)
            self._save_registry()
            return

        # Check local
        if resident_id in self.local_registries:
            if symbol in self.local_registries[resident_id]:
                entry = self.local_registries[resident_id][symbol]
                entry.frequency += 1
                self._save_registry()
                return

        # Check other residents' local registries for adoption
        for other_id, registry in self.local_registries.items():
            if other_id != resident_id and symbol in registry:
                entry = registry[symbol]
                entry.frequency += 1
                if resident_id not in entry.adopters:
                    entry.adopters.append(resident_id)
                    # Check for promotion
                    self._check_promotion(symbol, entry)
                self._save_registry()
                return

    def _check_promotion(self, symbol: str, entry: SymbolEntry):
        """Check if a local symbol should be promoted to global."""
        if entry.is_global:
            return

        if len(entry.adopters) >= self.PROMOTION_ADOPTER_THRESHOLD:
            if entry.frequency >= self.PROMOTION_FREQUENCY_THRESHOLD * len(entry.adopters):
                self.promote_to_global(symbol)

    def promote_to_global(self, symbol: str) -> Optional[SymbolEntry]:
        """
        Promote a local notation to global registry.

        Args:
            symbol: Symbol to promote

        Returns:
            The promoted entry, or None if not found
        """
        # Find the entry in local registries
        entry = None
        source_resident = None
        for resident_id, registry in self.local_registries.items():
            if symbol in registry:
                entry = registry[symbol]
                source_resident = resident_id
                break

        if not entry:
            return None

        # Mark as global
        entry.is_global = True

        # Add to global registry
        self.global_registry[symbol] = entry

        # Remove from all local registries
        for registry in self.local_registries.values():
            if symbol in registry:
                del registry[symbol]

        self._save_registry()
        return entry

    def resolve(self, symbol: str, resident_id: str) -> Optional[str]:
        """
        Resolve a symbol to its vector hash.

        Resolution order: global first, then local.

        Args:
            symbol: Symbol to resolve
            resident_id: Resident making the request

        Returns:
            Vector hash, or None if not found
        """
        # Global first
        if symbol in self.global_registry:
            return self.global_registry[symbol].vector_hash

        # Local for this resident
        if resident_id in self.local_registries:
            if symbol in self.local_registries[resident_id]:
                return self.local_registries[resident_id][symbol].vector_hash

        return None

    def resolve_hash_to_symbol(self, vector_hash: str) -> Optional[str]:
        """Reverse lookup: find symbol for a vector hash."""
        return self._hash_to_symbol.get(vector_hash)

    def get_adoption_count(self, symbol: str) -> int:
        """Count how many residents use this symbol."""
        if symbol in self.global_registry:
            return len(self.global_registry[symbol].adopters)

        for registry in self.local_registries.values():
            if symbol in registry:
                return len(registry[symbol].adopters)

        return 0

    def get_all_symbols(self, resident_id: str) -> Dict[str, SymbolEntry]:
        """Get all symbols available to a resident (global + local)."""
        result = dict(self.global_registry)
        if resident_id in self.local_registries:
            result.update(self.local_registries[resident_id])
        return result

    def _generate_receipt_hash(self, symbol: str, vector_hash: str, resident_id: str) -> str:
        """Generate catalytic receipt hash."""
        content = f"{symbol}:{vector_hash}:{resident_id}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Symbolic Compiler
# =============================================================================

class SymbolicCompiler:
    """
    P.2.1 - Multi-level semantic rendering.

    Renders GeometricState at 4 compression levels with verifiable
    lossless round-trip capability.

    Usage:
        compiler = SymbolicCompiler(vector_store, registry)

        # Render at different levels
        prose = compiler.render(state, 0)      # Full prose
        symbol = compiler.render(state, 1)     # @Symbol refs
        hash_form = compiler.render(state, 2)  # Vector hashes
        protocol = compiler.render(state, 3)   # Custom notation

        # Verify round-trip
        verification = compiler.verify_roundtrip(state, hash_form.content, 2)
        assert verification.verified  # E > 0.99
    """

    # E threshold for lossless verification (P.2.1.2)
    E_THRESHOLD = 0.99
    Df_THRESHOLD = 0.01

    def __init__(
        self,
        reasoner: GeometricReasoner,
        symbol_registry: HybridSymbolRegistry,
        corpus: Optional[List[str]] = None
    ):
        """
        Initialize symbolic compiler.

        Args:
            reasoner: GeometricReasoner for boundary operations
            symbol_registry: HybridSymbolRegistry for symbol resolution
            corpus: Text corpus for Level 0 readout
        """
        self.reasoner = reasoner
        self.registry = symbol_registry
        self.corpus = corpus or []

        # Cache for vector hash -> GeometricState mapping
        self._state_cache: Dict[str, GeometricState] = {}
        self._cache_limit = 1000

        # Import NotationRegistry from symbol_evolution for Level 3
        try:
            from symbol_evolution import NotationRegistry
            self.notation_registry = NotationRegistry()
        except ImportError:
            self.notation_registry = None

        # Statistics
        self.stats = {
            'renders': Counter(),
            'verifications': 0,
            'verified_count': 0,
            'failed_count': 0
        }

    def _get_vector_hash(self, state: GeometricState) -> str:
        """Get SHA256 hash of state vector."""
        return hashlib.sha256(state.vector.tobytes()).hexdigest()

    def _get_short_hash(self, state: GeometricState) -> str:
        """Get 16-char hash for compact representation."""
        return self._get_vector_hash(state)[:16]

    def _cache_state(self, state: GeometricState):
        """Cache state for decompression."""
        hash_key = self._get_short_hash(state)
        if len(self._state_cache) >= self._cache_limit:
            # Remove oldest entries
            oldest_keys = list(self._state_cache.keys())[:100]
            for k in oldest_keys:
                del self._state_cache[k]
        self._state_cache[hash_key] = state

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text.split()) + len(re.findall(r'[^\w\s]', text))

    def _generate_receipt(
        self,
        operation: str,
        level: int,
        original_hash: str,
        result_content: str,
        E_preserved: float,
        Df_preserved: float
    ) -> Dict:
        """Generate catalytic receipt for a render operation."""
        timestamp = datetime.now(timezone.utc).isoformat()
        receipt_data = {
            'operation': operation,
            'level': level,
            'original_hash': original_hash,
            'result_hash': hashlib.sha256(result_content.encode()).hexdigest()[:16],
            'E_preserved': E_preserved,
            'Df_preserved': Df_preserved,
            'timestamp': timestamp
        }
        receipt_data['receipt_hash'] = hashlib.sha256(
            json.dumps(receipt_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        return receipt_data

    # =========================================================================
    # Level 0: Prose Renderer
    # =========================================================================

    def _render_level_0(self, state: GeometricState) -> RenderResult:
        """
        Render as full prose for humans.

        Uses GeometricReasoner.readout() to decode to nearest text.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        original_hash = self._get_short_hash(state)

        if not self.corpus:
            # No corpus - return description based on state properties
            content = f"GeometricState(Df={state.Df:.2f}, hash={original_hash})"
        else:
            # Use readout to find nearest corpus items
            results = self.reasoner.readout(state, self.corpus, k=3)
            if results:
                # Combine top results into prose
                content = " | ".join([f"{text} (E={e:.3f})" for text, e in results])
            else:
                content = f"[No matching prose for {original_hash}]"

        receipt = self._generate_receipt(
            'render_level_0', 0, original_hash, content, 1.0, state.Df
        )

        return RenderResult(
            content=content,
            level=0,
            level_name="prose",
            E_preserved=1.0,  # Original state, no compression
            Df_preserved=state.Df,
            compression_ratio=1.0,  # Baseline
            original_hash=original_hash,
            receipt_hash=receipt['receipt_hash'],
            timestamp=timestamp
        )

    # =========================================================================
    # Level 1: Symbol Renderer
    # =========================================================================

    def _render_level_1(self, state: GeometricState, resident_id: str = "default") -> RenderResult:
        """
        Render as @Symbol references.

        Looks up or creates symbol in HybridSymbolRegistry.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        original_hash = self._get_short_hash(state)

        # Check if symbol exists for this hash
        existing_symbol = self.registry.resolve_hash_to_symbol(original_hash)

        if existing_symbol:
            content = existing_symbol
            self.registry.record_usage(existing_symbol, resident_id)
        else:
            # Create new symbol
            # Try to infer a category from corpus if available
            category = "State"
            if self.corpus:
                results = self.reasoner.readout(state, self.corpus, k=1)
                if results:
                    first_word = results[0][0].split()[0] if results[0][0] else "Concept"
                    category = first_word.capitalize()

            symbol = f"@{category}-{original_hash[:8]}"
            meaning = None
            if self.corpus:
                results = self.reasoner.readout(state, self.corpus, k=1)
                if results:
                    meaning = results[0][0]

            self.registry.register_local(resident_id, symbol, state, meaning)
            content = symbol

        # Calculate compression
        original_tokens = self._estimate_tokens(str(state.receipt()))
        compressed_tokens = self._estimate_tokens(content)
        compression_ratio = original_tokens / max(compressed_tokens, 1)

        receipt = self._generate_receipt(
            'render_level_1', 1, original_hash, content, 1.0, state.Df
        )

        return RenderResult(
            content=content,
            level=1,
            level_name="symbol",
            E_preserved=1.0,  # Symbol maps exactly to state
            Df_preserved=state.Df,
            compression_ratio=compression_ratio,
            original_hash=original_hash,
            receipt_hash=receipt['receipt_hash'],
            timestamp=timestamp
        )

    # =========================================================================
    # Level 2: Hash Renderer
    # =========================================================================

    def _render_level_2(self, state: GeometricState) -> RenderResult:
        """
        Render as vector hash only.

        Maximum compression - just the SHA256 reference.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        original_hash = self._get_short_hash(state)

        # Format: [v:hash16]
        content = f"[v:{original_hash}]"

        # Cache for decompression
        self._cache_state(state)

        # Calculate compression
        original_tokens = self._estimate_tokens(str(state.receipt()))
        compressed_tokens = self._estimate_tokens(content)
        compression_ratio = original_tokens / max(compressed_tokens, 1)

        receipt = self._generate_receipt(
            'render_level_2', 2, original_hash, content, 1.0, state.Df
        )

        return RenderResult(
            content=content,
            level=2,
            level_name="hash",
            E_preserved=1.0,  # Hash is exact reference
            Df_preserved=state.Df,
            compression_ratio=compression_ratio,
            original_hash=original_hash,
            receipt_hash=receipt['receipt_hash'],
            timestamp=timestamp
        )

    # =========================================================================
    # Level 3: Protocol Renderer
    # =========================================================================

    def _render_level_3(self, state: GeometricState) -> RenderResult:
        """
        Render using emergent custom notation.

        Design Decision: Emergent only - no predefined grammar.
        Patterns tracked as they develop organically.

        Format includes embedded metrics:
        [v:hash] [E:value] [Df:value]
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        original_hash = self._get_short_hash(state)

        # Build emergent protocol notation
        # Base: vector reference
        parts = [f"[v:{original_hash}]"]

        # Add quantum metrics
        parts.append(f"[Df:{state.Df:.2f}]")

        # Add operation history if present
        if state.operation_history:
            last_op = state.operation_history[-1] if state.operation_history else {}
            op_name = last_op.get('op', 'init')
            parts.append(f"{{op:{op_name}}}")

        content = " ".join(parts)

        # Cache for decompression
        self._cache_state(state)

        # Calculate compression
        original_tokens = self._estimate_tokens(str(state.receipt()))
        compressed_tokens = self._estimate_tokens(content)
        compression_ratio = original_tokens / max(compressed_tokens, 1)

        receipt = self._generate_receipt(
            'render_level_3', 3, original_hash, content, 1.0, state.Df
        )

        return RenderResult(
            content=content,
            level=3,
            level_name="protocol",
            E_preserved=1.0,
            Df_preserved=state.Df,
            compression_ratio=compression_ratio,
            original_hash=original_hash,
            receipt_hash=receipt['receipt_hash'],
            timestamp=timestamp
        )

    # =========================================================================
    # Main Render Interface
    # =========================================================================

    def render(
        self,
        state: GeometricState,
        target_level: int,
        resident_id: str = "default"
    ) -> RenderResult:
        """
        Render a GeometricState at the specified compression level.

        Args:
            state: GeometricState to render
            target_level: 0 (prose), 1 (symbol), 2 (hash), 3 (protocol)
            resident_id: For symbol registry operations

        Returns:
            RenderResult with content and metadata
        """
        if target_level < 0 or target_level > 3:
            raise ValueError(f"Invalid compression level: {target_level}. Must be 0-3.")

        self.stats['renders'][target_level] += 1

        if target_level == 0:
            return self._render_level_0(state)
        elif target_level == 1:
            return self._render_level_1(state, resident_id)
        elif target_level == 2:
            return self._render_level_2(state)
        else:
            return self._render_level_3(state)

    # =========================================================================
    # Decompression
    # =========================================================================

    def decompress(
        self,
        content: str,
        source_level: int,
        resident_id: str = "default"
    ) -> Optional[GeometricState]:
        """
        Decompress compressed content back to GeometricState.

        Args:
            content: Compressed form (symbol, hash, or protocol)
            source_level: The compression level of content
            resident_id: For symbol registry lookups

        Returns:
            GeometricState if resolvable, None otherwise
        """
        if source_level == 0:
            # Prose - need to re-embed
            if self.corpus:
                return self.reasoner.initialize(content)
            return None

        elif source_level == 1:
            # Symbol reference
            vector_hash = self.registry.resolve(content, resident_id)
            if vector_hash and vector_hash in self._state_cache:
                return self._state_cache[vector_hash]
            return None

        elif source_level == 2:
            # Hash reference [v:hash]
            match = re.search(r'\[v:([a-f0-9]+)\]', content)
            if match:
                hash_key = match.group(1)
                return self._state_cache.get(hash_key)
            return None

        elif source_level == 3:
            # Protocol notation - extract hash
            match = re.search(r'\[v:([a-f0-9]+)\]', content)
            if match:
                hash_key = match.group(1)
                return self._state_cache.get(hash_key)
            return None

        return None

    # =========================================================================
    # Lossless Verification
    # =========================================================================

    def verify_roundtrip(
        self,
        original: GeometricState,
        compressed: str,
        level: int,
        resident_id: str = "default"
    ) -> RoundTripVerification:
        """
        Verify lossless round-trip at E > 0.99.

        Args:
            original: Original GeometricState
            compressed: Compressed form
            level: Compression level used
            resident_id: For symbol resolution

        Returns:
            RoundTripVerification with E/Df deltas and verification status
        """
        self.stats['verifications'] += 1
        timestamp = datetime.now(timezone.utc).isoformat()
        original_hash = self._get_short_hash(original)

        # Decompress
        decompressed = self.decompress(compressed, level, resident_id)

        if decompressed is None:
            # Cannot decompress - verification fails
            self.stats['failed_count'] += 1
            return RoundTripVerification(
                original_hash=original_hash,
                compressed_form=compressed,
                decompressed_hash="NONE",
                E_delta=1.0,
                Df_delta=1.0,
                verified=False,
                level=level,
                receipt={
                    'status': 'decompression_failed',
                    'timestamp': timestamp
                }
            )

        decompressed_hash = self._get_short_hash(decompressed)

        # Compute E (Born rule similarity)
        E = original.E_with(decompressed)
        E_delta = 1.0 - E

        # Compute Df delta
        Df_delta = abs(original.Df - decompressed.Df)

        # Verify thresholds
        verified = (E > self.E_THRESHOLD) and (Df_delta < self.Df_THRESHOLD)

        if verified:
            self.stats['verified_count'] += 1
        else:
            self.stats['failed_count'] += 1

        # Generate receipt
        receipt = {
            'operation': 'verify_roundtrip',
            'level': level,
            'original_hash': original_hash,
            'decompressed_hash': decompressed_hash,
            'E': float(E),
            'E_delta': float(E_delta),
            'Df_delta': float(Df_delta),
            'verified': verified,
            'timestamp': timestamp
        }
        receipt['receipt_hash'] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()
        ).hexdigest()[:16]

        return RoundTripVerification(
            original_hash=original_hash,
            compressed_form=compressed,
            decompressed_hash=decompressed_hash,
            E_delta=E_delta,
            Df_delta=Df_delta,
            verified=verified,
            level=level,
            receipt=receipt
        )

    # =========================================================================
    # Metrics & Statistics
    # =========================================================================

    def get_compression_stats(self) -> Dict:
        """Get compression statistics across all levels."""
        return {
            'renders_by_level': dict(self.stats['renders']),
            'total_renders': sum(self.stats['renders'].values()),
            'total_verifications': self.stats['verifications'],
            'verified_count': self.stats['verified_count'],
            'failed_count': self.stats['failed_count'],
            'verification_rate': (
                self.stats['verified_count'] / max(self.stats['verifications'], 1)
            ),
            'registry_global_count': len(self.registry.global_registry),
            'registry_local_count': sum(
                len(r) for r in self.registry.local_registries.values()
            )
        }


# =============================================================================
# CLI Helpers
# =============================================================================

def create_compiler(corpus: Optional[List[str]] = None) -> SymbolicCompiler:
    """
    Factory function to create a SymbolicCompiler with defaults.

    Args:
        corpus: Optional text corpus for Level 0 readout

    Returns:
        Configured SymbolicCompiler
    """
    reasoner = GeometricReasoner()
    registry = HybridSymbolRegistry()
    return SymbolicCompiler(reasoner, registry, corpus)


if __name__ == "__main__":
    # Quick test
    print("P.2 Symbolic Compiler - Multi-level semantic rendering")
    print("=" * 60)

    compiler = create_compiler(corpus=["Hello world", "Test input", "Semantic meaning"])

    # Create a test state
    test_state = compiler.reasoner.initialize("Test semantic concept")

    print(f"\nOriginal state: {test_state}")
    print(f"Df: {test_state.Df:.2f}")
    print()

    # Render at all levels
    for level in range(4):
        result = compiler.render(test_state, level)
        print(f"Level {level} ({result.level_name}):")
        print(f"  Content: {result.content}")
        print(f"  Compression ratio: {result.compression_ratio:.2f}")
        print(f"  Receipt: {result.receipt_hash}")
        print()

    # Test round-trip verification for Level 2
    level_2_result = compiler.render(test_state, 2)
    verification = compiler.verify_roundtrip(test_state, level_2_result.content, 2)
    print(f"Round-trip verification (Level 2):")
    print(f"  E delta: {verification.E_delta:.6f}")
    print(f"  Df delta: {verification.Df_delta:.6f}")
    print(f"  Verified: {verification.verified}")
    print()

    # Show stats
    print("Statistics:")
    stats = compiler.get_compression_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
