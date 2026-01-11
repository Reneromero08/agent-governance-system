#!/usr/bin/env python3
"""
Symbol Resolver - Resolve @C symbols to canon content

Practical tool for agents to use compressed canon symbols.

Usage:
    from symbol_resolver import SymbolResolver

    resolver = SymbolResolver()
    content = resolver.resolve("@C:b0fc0ba6b82e")
    print(content)
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional


class SymbolResolver:
    """Resolves @C symbols to canon file content"""

    def __init__(self, repo_root: str = None):
        if repo_root is None:
            # Detect repo root
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / "LAW" / "CANON").exists():
                    repo_root = str(current)
                    break
                current = current.parent
            else:
                raise RuntimeError("Could not find repo root (no LAW/CANON)")

        self.repo_root = Path(repo_root)
        self.symbol_table_path = (
            self.repo_root / "MEMORY" / "LLM_PACKER" / "_compressed" / "canon_symbol_table.json"
        )

        # Load symbol table
        self._symbol_table: Optional[Dict] = None

    def _load_symbol_table(self):
        """Lazy load symbol table"""
        if self._symbol_table is None:
            if not self.symbol_table_path.exists():
                raise FileNotFoundError(
                    f"Symbol table not found at {self.symbol_table_path}. "
                    "Run: python CAPABILITY/PRIMITIVES/canon_compressor.py --compress"
                )

            with open(self.symbol_table_path, 'r', encoding='utf-8') as f:
                self._symbol_table = json.load(f)

    def resolve(self, symbol: str, verify: bool = True) -> str:
        """
        Resolve symbol to file content

        Args:
            symbol: @C:hash_short symbol
            verify: Verify SHA-256 hash after reading

        Returns:
            File content as string

        Raises:
            KeyError: Symbol not found
            FileNotFoundError: File doesn't exist
            ValueError: Hash verification failed
        """
        self._load_symbol_table()

        if symbol not in self._symbol_table:
            raise KeyError(f"Symbol {symbol} not found in table")

        entry = self._symbol_table[symbol]
        file_path = self.repo_root / entry['path']

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read content
        content = file_path.read_text(encoding='utf-8')

        # Verify hash if requested
        if verify:
            computed_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            expected_hash = entry['sha256']

            if computed_hash != expected_hash:
                raise ValueError(
                    f"Hash verification failed for {symbol}\n"
                    f"Expected: {expected_hash}\n"
                    f"Computed: {computed_hash}"
                )

        return content

    def get_path(self, symbol: str) -> str:
        """Get file path for symbol (without reading content)"""
        self._load_symbol_table()

        if symbol not in self._symbol_table:
            raise KeyError(f"Symbol {symbol} not found in table")

        return self._symbol_table[symbol]['path']

    def get_metadata(self, symbol: str) -> Dict:
        """Get metadata for symbol"""
        self._load_symbol_table()

        if symbol not in self._symbol_table:
            raise KeyError(f"Symbol {symbol} not found in table")

        return self._symbol_table[symbol].copy()

    def list_symbols(self) -> Dict[str, str]:
        """List all available symbols with paths"""
        self._load_symbol_table()
        return {
            symbol: entry['path']
            for symbol, entry in self._symbol_table.items()
        }


def main():
    """CLI interface for symbol resolution"""
    import argparse

    parser = argparse.ArgumentParser(description="Resolve @C symbols to canon content")
    parser.add_argument('symbol', nargs='?', help="Symbol to resolve (e.g., @C:b0fc0ba6b82e)")
    parser.add_argument('--list', action='store_true', help="List all available symbols")
    parser.add_argument('--path-only', action='store_true', help="Show only file path")
    parser.add_argument('--metadata', action='store_true', help="Show metadata only")
    parser.add_argument('--no-verify', action='store_true', help="Skip hash verification")

    args = parser.parse_args()

    resolver = SymbolResolver()

    if args.list:
        print("Available symbols:")
        for symbol, path in sorted(resolver.list_symbols().items()):
            print(f"  {symbol} → {path}")
        return

    if not args.symbol:
        parser.print_help()
        return

    try:
        if args.path_only:
            print(resolver.get_path(args.symbol))
        elif args.metadata:
            metadata = resolver.get_metadata(args.symbol)
            print(json.dumps(metadata, indent=2))
        else:
            content = resolver.resolve(args.symbol, verify=not args.no_verify)
            print(content)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
