#!/usr/bin/env python3
"""
Symbol Resolver

Resolves symbols to bounded section content with caching.

Roadmap Phase: Phase 2.2 — Symbol resolution + expansion cache
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from catalytic_chat.section_indexer import SectionIndexer
from catalytic_chat.slice_resolver import SliceResolver, SliceError
from catalytic_chat.symbol_registry import SymbolRegistry, SymbolError


@dataclass
class ExpansionCacheEntry:
    """Cache entry for symbol expansion."""
    run_id: str
    symbol_id: str
    slice: str
    section_id: str
    section_content_hash: str
    payload: str
    payload_hash: str
    bytes_expanded: int
    created_at: str


class ResolverError(Exception):
    """Symbol resolution error."""
    pass


class SymbolResolver:
    """Resolves symbols to bounded section content with caching."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        substrate_mode: str = "sqlite",
        symbol_registry: Optional['SymbolRegistry'] = None
    ):
        """Initialize symbol resolver.

        Args:
            repo_root: Repository root path
            substrate_mode: "sqlite" or "jsonl"
            symbol_registry: SymbolRegistry instance (must be provided)
        """
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = Path(repo_root)
        self.substrate_mode = substrate_mode
        self.slice_resolver = SliceResolver()
        self.section_indexer = SectionIndexer(repo_root, substrate_mode)
        self.symbol_registry = symbol_registry

        if substrate_mode == "sqlite":
            self.db_path = repo_root / "CORTEX" / "db" / "system1.db"
        elif substrate_mode == "jsonl":
            self.cache_path = repo_root / "CORTEX" / "_generated" / "expansion_cache.jsonl"
        else:
            raise ValueError(f"Invalid substrate_mode: {substrate_mode}")

    def _get_timestamp(self) -> str:
        """Get ISO8601 timestamp.

        Returns:
            ISO8601 string with timezone
        """
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def _compute_payload_hash(self, payload: str) -> str:
        """Compute SHA-256 hash of payload.

        Args:
            payload: Payload content

        Returns:
            SHA-256 hash as hex string
        """
        normalized = payload.replace('\r\n', '\n')
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _cache_get_sqlite(
        self,
        run_id: str,
        symbol_id: str,
        slice_expr: str,
        section_id: str
    ) -> Optional[ExpansionCacheEntry]:
        """Get cache entry from SQLite.

        Args:
            run_id: Run ID
            symbol_id: Symbol ID
            slice_expr: Slice expression
            section_id: Section ID

        Returns:
            Cache entry or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM expansion_cache
                WHERE run_id = ? AND symbol_id = ? AND slice_expr = ? AND section_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (run_id, symbol_id, slice_expr, section_id))

            row = cursor.fetchone()
            if row:
                return ExpansionCacheEntry(
                    run_id=row['run_id'],
                    symbol_id=row['symbol_id'],
                    slice=row['slice_expr'],
                    section_id=row['section_id'],
                    section_content_hash=row['section_content_hash'],
                    payload=row['payload'],
                    payload_hash=row['payload_hash'],
                    bytes_expanded=row['bytes_expanded'],
                    created_at=row['created_at']
                )
            return None

    def _cache_get_jsonl(
        self,
        run_id: str,
        symbol_id: str,
        slice_expr: str,
        section_id: str
    ) -> Optional[ExpansionCacheEntry]:
        """Get cache entry from JSONL.

        Args:
            run_id: Run ID
            symbol_id: Symbol ID
            slice_expr: Slice expression
            section_id: Section ID

        Returns:
            Cache entry or None if not found
        """
        if not self.cache_path.exists():
            return None

        with open(self.cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if (entry['run_id'] == run_id and
                    entry['symbol_id'] == symbol_id and
                    entry['slice'] == slice_expr and
                    entry['section_id'] == section_id):
                    return ExpansionCacheEntry(**entry)
        return None

    def _cache_put_sqlite(self, entry: ExpansionCacheEntry) -> None:
        """Put cache entry into SQLite.

        Args:
            entry: Cache entry to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS expansion_cache (
                    run_id TEXT NOT NULL,
                    symbol_id TEXT NOT NULL,
                    slice TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    section_content_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    bytes_expanded INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, symbol_id, slice, section_id),
                    FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE CASCADE
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_run ON expansion_cache(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_created ON expansion_cache(created_at)")

            conn.execute("""
                INSERT OR REPLACE INTO expansion_cache (
                    run_id, symbol_id, slice, section_id,
                    section_content_hash, payload, payload_hash,
                    bytes_expanded, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.run_id,
                entry.symbol_id,
                entry.slice,
                entry.section_id,
                entry.section_content_hash,
                entry.payload,
                entry.payload_hash,
                entry.bytes_expanded,
                entry.created_at
            ))

            conn.commit()

    def _cache_put_jsonl(self, entry: ExpansionCacheEntry) -> None:
        """Put cache entry into JSONL.

        Args:
            entry: Cache entry to store
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def resolve(
        self,
        symbol_id: str,
        slice_expr: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Resolve symbol to payload.

        Args:
            symbol_id: Symbol ID to resolve
            slice_expr: Optional slice expression
            run_id: Optional run ID for caching

        Returns:
            Tuple of (payload, cache_hit)

        Raises:
            ResolverError: If symbol not found or resolution fails
        """
        symbol = self.symbol_registry.get_symbol(symbol_id)

        if symbol is None:
            raise ResolverError(f"Symbol not found: {symbol_id}")

        if symbol.target_type != "SECTION":
            raise ResolverError(f"Unsupported target type: {symbol.target_type}")

        section_id = symbol.target_ref

        if slice_expr is None:
            slice_expr = symbol.default_slice if symbol.default_slice else "none"

        content, section_hash, applied_slice, lines_applied, chars_applied = \
            self.section_indexer.get_section_content(section_id, slice_expr)

        payload_hash = self._compute_payload_hash(content)

        cache_hit = False
        if run_id is not None:
            cache_key = (run_id, symbol_id, slice_expr, section_id, section_hash)

            if self.substrate_mode == "sqlite":
                cached = self._cache_get_sqlite(*cache_key)
                if cached:
                    if cached.payload_hash == payload_hash:
                        content = cached.payload
                        cache_hit = True
                    else:
                        self._cache_put_sqlite(ExpansionCacheEntry(
                            run_id=run_id,
                            symbol_id=symbol_id,
                            slice=slice_expr,
                            section_id=section_id,
                            section_content_hash=section_hash,
                            payload=content,
                            payload_hash=payload_hash,
                            bytes_expanded=len(content.encode('utf-8')),
                            created_at=self._get_timestamp()
                        ))
            else:
                cached = self._cache_get_jsonl(*cache_key)
                if cached and cached.payload_hash == payload_hash:
                    content = cached.payload
                    cache_hit = True
                elif cached is None or cached.payload_hash != payload_hash:
                    self._cache_put_jsonl(ExpansionCacheEntry(
                        run_id=run_id,
                        symbol_id=symbol_id,
                        slice=slice_expr,
                        section_id=section_id,
                        section_content_hash=section_hash,
                        payload=content,
                        payload_hash=payload_hash,
                        bytes_expanded=len(content.encode('utf-8')),
                        created_at=self._get_timestamp()
                    ))

        return content, cache_hit


def resolve_symbol(
    repo_root: Optional[Path] = None,
    substrate_mode: str = "sqlite",
    symbol_id: str = "",
    slice_expr: Optional[str] = None,
    run_id: Optional[str] = None
) -> Tuple[str, bool]:
    """Convenience function to resolve symbol.

    Args:
        repo_root: Repository root path
        substrate_mode: Substrate mode
        symbol_id: Symbol ID to resolve
        slice_expr: Optional slice expression
        run_id: Optional run ID for caching

    Returns:
        Tuple of (payload, cache_hit)

    Raises:
        ResolverError: If symbol not found or resolution fails
    """
    from .symbol_registry import SymbolRegistry

    symbol_registry = SymbolRegistry(repo_root, substrate_mode)
    resolver = SymbolResolver(repo_root, substrate_mode, symbol_registry)
    return resolver.resolve(symbol_id, slice_expr, run_id)


if __name__ == '__main__':
    import sys
    import uuid

    resolver = SymbolResolver()

    run_id = str(uuid.uuid4())

    test_symbol = "@TEST/example"

    print(f"Testing resolve with run_id: {run_id}")

    try:
        payload1, hit1 = resolver.resolve(test_symbol, run_id=run_id)
        print(f"First resolve: hit={hit1}")
        print(f"Payload length: {len(payload1)} chars")

        payload2, hit2 = resolver.resolve(test_symbol, run_id=run_id)
        print(f"Second resolve: hit={hit2}")
        print(f"Payload length: {len(payload2)} chars")

        if hit1 and hit2:
            print("✓ Cache working correctly (both hits)")
        elif not hit1 and hit2:
            print("✓ Cache working correctly (second is hit)")
        else:
            print("✗ Cache not working (both misses)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
