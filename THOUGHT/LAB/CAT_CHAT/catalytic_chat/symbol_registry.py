#!/usr/bin/env python3
"""
Symbol Registry

Manages SYMBOLS artifact mapping symbol IDs to section IDs with deterministic ordering.

Roadmap Phase: Phase 2.1 — Symbol registry
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from catalytic_chat.section_indexer import SectionIndexer
from catalytic_chat.slice_resolver import SliceResolver, SliceError
from .paths import get_cortex_dir, get_system1_db, get_sqlite_connection


@dataclass
class Symbol:
    """Symbol entry in registry."""
    symbol_id: str
    target_type: str
    target_ref: str
    default_slice: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class SymbolError(Exception):
    """Symbol registry error."""
    pass


class SymbolRegistry:
    """Manages symbol registry with dual substrate support."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        substrate_mode: str = "sqlite"
    ):
        """Initialize symbol registry.

        Args:
            repo_root: Repository root path. Defaults to current working directory.
            substrate_mode: "sqlite" or "jsonl"
        """
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = Path(repo_root)
        self.substrate_mode = substrate_mode
        self.slice_resolver = SliceResolver()

        if substrate_mode == "sqlite":
            self.db_path = get_system1_db(self.repo_root)
        elif substrate_mode == "jsonl":
            cortex_dir = get_cortex_dir(self.repo_root)
            self.output_path = cortex_dir / "symbols.jsonl"
        else:
            raise ValueError(f"Invalid substrate_mode: {substrate_mode}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite symbols table."""
        with get_sqlite_connection(self.db_path) as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_ref TEXT NOT NULL,
                    default_slice TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (target_ref) REFERENCES sections(section_id) ON DELETE CASCADE
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_target ON symbols(target_ref)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_created ON symbols(created_at)")

            conn.commit()

    def _get_timestamp(self) -> str:
        """Get ISO8601 timestamp.

        Returns:
            ISO8601 string with timezone
        """
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def _validate_symbol_id(self, symbol_id: str) -> None:
        """Validate symbol ID format.

        Args:
            symbol_id: Symbol ID to validate

        Raises:
            SymbolError: If invalid format
        """
        if not symbol_id:
            raise SymbolError("Symbol ID cannot be empty")

        if not symbol_id.startswith('@'):
            raise SymbolError(f"Symbol ID must start with '@': {symbol_id}")

    def _validate_target_ref(self, target_ref: str) -> None:
        """Validate that target_ref exists in SECTION_INDEX.

        Args:
            target_ref: Section ID

        Raises:
            SymbolError: If section not found
        """
        indexer = SectionIndexer(self.repo_root, self.substrate_mode)
        section = indexer.get_section_by_id(target_ref)

        if section is None:
            raise SymbolError(f"Target section not found in SECTION_INDEX: {target_ref}")

    def _validate_default_slice(self, default_slice: Optional[str]) -> None:
        """Validate default slice expression.

        Args:
            default_slice: Slice expression

        Raises:
            SymbolError: If slice is invalid
        """
        if default_slice is None:
            return

        try:
            self.slice_resolver.parse_slice(default_slice, 1000, 10000)
        except SliceError as e:
            raise SymbolError(f"Invalid default slice '{default_slice}': {e}")

        if default_slice.lower() == "all":
            raise SymbolError(f"Default slice cannot be 'ALL' (unbounded expansion forbidden)")

    def add_symbol(
        self,
        symbol_id: str,
        target_ref: str,
        default_slice: Optional[str] = None
    ) -> str:
        """Add symbol to registry.

        Args:
            symbol_id: Symbol ID (must start with '@')
            target_ref: Section ID
            default_slice: Optional default slice expression

        Returns:
            Timestamp of creation

        Raises:
            SymbolError: If validation fails
        """
        self._validate_symbol_id(symbol_id)
        self._validate_target_ref(target_ref)
        self._validate_default_slice(default_slice)

        timestamp = self._get_timestamp()

        if self.substrate_mode == "sqlite":
            self._add_symbol_sqlite(symbol_id, target_ref, default_slice, timestamp)
        else:
            self._add_symbol_jsonl(symbol_id, target_ref, default_slice, timestamp)

        return timestamp

    def _add_symbol_sqlite(
        self,
        symbol_id: str,
        target_ref: str,
        default_slice: Optional[str],
        timestamp: str
    ) -> None:
        """Add symbol to SQLite registry.

        Args:
            symbol_id: Symbol ID
            target_ref: Section ID
            default_slice: Default slice
            timestamp: ISO8601 timestamp
        """
        with get_sqlite_connection(self.db_path) as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_ref TEXT NOT NULL,
                    default_slice TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (target_ref) REFERENCES sections(section_id) ON DELETE CASCADE
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_target ON symbols(target_ref)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_created ON symbols(created_at)")

            cursor = conn.execute(
                "SELECT symbol_id FROM symbols WHERE symbol_id = ?",
                (symbol_id,)
            )
            if cursor.fetchone():
                raise SymbolError(f"Symbol ID already exists: {symbol_id}")

            conn.execute("""
                INSERT INTO symbols (
                    symbol_id, target_type, target_ref, default_slice, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol_id, "SECTION", target_ref, default_slice, timestamp, timestamp))

            conn.commit()

        print(f"Added symbol: {symbol_id}")

    def _add_symbol_jsonl(
        self,
        symbol_id: str,
        target_ref: str,
        default_slice: Optional[str],
        timestamp: str
    ) -> None:
        """Add symbol to JSONL registry.

        Args:
            symbol_id: Symbol ID
            target_ref: Section ID
            default_slice: Default slice
            timestamp: ISO8601 timestamp
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_symbols = set()
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    existing_symbols.add(data['symbol_id'])

        if symbol_id in existing_symbols:
            raise SymbolError(f"Symbol ID already exists: {symbol_id}")

        with open(self.output_path, 'a', encoding='utf-8') as f:
            symbol = {
                'symbol_id': symbol_id,
                'target_type': 'SECTION',
                'target_ref': target_ref,
                'default_slice': default_slice,
                'created_at': timestamp,
                'updated_at': timestamp
            }

            f.write(json.dumps(symbol) + '\n')

        print(f"Added symbol: {symbol_id}")

    def get_symbol(self, symbol_id: str) -> Optional[Symbol]:
        """Get symbol by ID.

        Args:
            symbol_id: Symbol ID

        Returns:
            Symbol object or None if not found
        """
        if self.substrate_mode == "sqlite":
            return self._get_symbol_sqlite(symbol_id)
        else:
            return self._get_symbol_jsonl(symbol_id)

    def _get_symbol_sqlite(self, symbol_id: str) -> Optional[Symbol]:
        """Get symbol from SQLite registry.

        Args:
            symbol_id: Symbol ID

        Returns:
            Symbol object or None
        """
        try:
            with get_sqlite_connection(self.db_path) as conn:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS symbols (
                        symbol_id TEXT PRIMARY KEY,
                        target_type TEXT NOT NULL,
                        target_ref TEXT NOT NULL,
                        default_slice TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (target_ref) REFERENCES sections(section_id) ON DELETE CASCADE
                    )
                """)

                cursor = conn.execute(
                    "SELECT * FROM symbols WHERE symbol_id = ?",
                    (symbol_id,)
                )
                row = cursor.fetchone()

                if row:
                    return Symbol(
                        symbol_id=row['symbol_id'],
                        target_type=row['target_type'],
                        target_ref=row['target_ref'],
                        default_slice=row['default_slice'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                return None
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return None
            raise

    def _get_symbol_jsonl(self, symbol_id: str) -> Optional[Symbol]:
        """Get symbol from JSONL registry.

        Args:
            symbol_id: Symbol ID

        Returns:
            Symbol object or None
        """
        if not self.output_path.exists():
            return None

        with open(self.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['symbol_id'] == symbol_id:
                    return Symbol(**data)
        return None

    def list_symbols(self, prefix: Optional[str] = None) -> List[Symbol]:
        """List all symbols, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter (e.g., "@CANON/")

        Returns:
            List of symbols, sorted by symbol_id for determinism
        """
        if self.substrate_mode == "sqlite":
            return self._list_symbols_sqlite(prefix)
        else:
            return self._list_symbols_jsonl(prefix)

    def _list_symbols_sqlite(self, prefix: Optional[str]) -> List[Symbol]:
        """List symbols from SQLite registry.

        Args:
            prefix: Optional prefix filter

        Returns:
            List of symbols sorted by symbol_id
        """
        with get_sqlite_connection(self.db_path) as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_ref TEXT NOT NULL,
                    default_slice TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (target_ref) REFERENCES sections(section_id) ON DELETE CASCADE
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_target ON symbols(target_ref)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_created ON symbols(created_at)")

            cursor = conn.execute("SELECT * FROM symbols ORDER BY symbol_id")
            rows = cursor.fetchall()

            symbols = []
            for row in rows:
                if prefix is None or row['symbol_id'].startswith(prefix):
                    symbols.append(Symbol(
                        symbol_id=row['symbol_id'],
                        target_type=row['target_type'],
                        target_ref=row['target_ref'],
                        default_slice=row['default_slice'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    ))

            return symbols

    def _list_symbols_jsonl(self, prefix: Optional[str]) -> List[Symbol]:
        """List symbols from JSONL registry.

        Args:
            prefix: Optional prefix filter

        Returns:
            List of symbols sorted by symbol_id
        """
        if not self.output_path.exists():
            return []

        symbols = []
        with open(self.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if prefix is None or data['symbol_id'].startswith(prefix):
                    symbols.append(Symbol(**data))

        return sorted(symbols, key=lambda x: x.symbol_id)

    def verify(self) -> bool:
        """Verify symbol registry integrity.

        Returns:
            True if verification passes, False otherwise
        """
        print("Verifying symbol registry...")

        try:
            symbols = self.list_symbols()

            if len(symbols) == 0:
                print("[OK] No symbols in registry")
                return True

            errors = 0
            for symbol in symbols:
                if not symbol.symbol_id.startswith('@'):
                    print(f"[ERROR] Invalid symbol_id format: {symbol.symbol_id}")
                    errors += 1

                if symbol.target_type != "SECTION":
                    print(f"[ERROR] Invalid target_type: {symbol.target_type}")
                    errors += 1

                section = SectionIndexer(self.repo_root, self.substrate_mode).get_section_by_id(symbol.target_ref)
                if section is None:
                    print(f"[ERROR] Target section not found: {symbol.target_ref}")
                    errors += 1

                if symbol.default_slice:
                    try:
                        self.slice_resolver.parse_slice(symbol.default_slice, 1000, 10000)
                    except SliceError as e:
                        print(f"[ERROR] Invalid default slice: {e}")
                        errors += 1

                    if symbol.default_slice.lower() == "all":
                        print(f"[ERROR] Default slice cannot be 'ALL': {symbol.symbol_id}")
                        errors += 1

            if errors == 0:
                print(f"[OK] Verified {len(symbols)} symbols")
                return True
            else:
                print(f"[FAIL] Found {errors} errors")
                return False

        except Exception as e:
            print(f"[FAIL] Verification error: {e}")
            return False


def add_symbol(
    repo_root: Optional[Path] = None,
    substrate_mode: str = "sqlite",
    symbol_id: str = "",
    target_ref: str = "",
    default_slice: Optional[str] = None
) -> str:
    """Convenience function to add a symbol.

    Args:
        repo_root: Repository root path
        substrate_mode: Substrate mode
        symbol_id: Symbol ID
        target_ref: Section ID
        default_slice: Default slice

    Returns:
        Timestamp of creation
    """
    registry = SymbolRegistry(repo_root, substrate_mode)
    return registry.add_symbol(symbol_id, target_ref, default_slice)


if __name__ == '__main__':
    import sys

    registry = SymbolRegistry()
    registry.verify()
