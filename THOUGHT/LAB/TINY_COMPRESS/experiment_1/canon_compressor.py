#!/usr/bin/env python3
"""
Canon Compressor - Practical H(X|S) compression for AGS Canon

Implements symbol-based compression where:
- H(X|S) = H(X) - I(X;S)
- When I(X;S) ≈ H(X), then H(X|S) ≈ 0

This tool:
1. Scans LAW/CANON/ for all .md files
2. Computes content hashes (sha256)
3. Generates @C symbols for each file
4. Creates a compressed manifest with symbols instead of full content
5. Measures compression ratio

Usage:
    python canon_compressor.py --compress
    python canon_compressor.py --decompress
    python canon_compressor.py --stats
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CanonFile:
    """Represents a canon file with metadata"""
    path: str
    size_bytes: int
    sha256: str
    symbol: str  # @C:{hash_short}
    content_preview: str  # First 100 chars for verification


@dataclass
class CompressionManifest:
    """Compressed representation of canon"""
    version: str
    canon_root: str
    total_files: int
    total_bytes: int
    compressed_bytes: int
    compression_ratio: float
    files: List[Dict]  # List of CanonFile dicts


class CanonCompressor:
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
        self.canon_dir = self.repo_root / "LAW" / "CANON"
        self.output_dir = self.repo_root / "MEMORY" / "LLM_PACKER" / "_compressed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_canon(self) -> List[CanonFile]:
        """Scan LAW/CANON and create symbol mappings"""
        canon_files = []

        for md_file in sorted(self.canon_dir.rglob("*.md")):
            # Read file
            content = md_file.read_text(encoding='utf-8')
            size = len(content.encode('utf-8'))

            # Compute hash
            sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest()
            hash_short = sha256[:12]  # First 12 chars for symbol

            # Relative path from CANON
            rel_path = str(md_file.relative_to(self.canon_dir))

            # Create symbol
            symbol = f"@C:{hash_short}"

            # Preview (first 100 chars, strip whitespace)
            preview = content[:100].strip().replace('\n', ' ')

            canon_files.append(CanonFile(
                path=rel_path,
                size_bytes=size,
                sha256=sha256,
                symbol=symbol,
                content_preview=preview
            ))

        return canon_files

    def compute_compression_ratio(self, canon_files: List[CanonFile]) -> Tuple[int, int, float]:
        """Compute compression ratio using symbol representation"""
        # Original size: sum of all file sizes
        total_bytes = sum(f.size_bytes for f in canon_files)

        # Compressed size: JSON manifest with symbols (not full content)
        manifest_dict = {
            "version": "1.0.0",
            "files": [
                {
                    "symbol": f.symbol,
                    "path": f.path,
                    "sha256": f.sha256,
                    "size": f.size_bytes,
                    "preview": f.content_preview
                }
                for f in canon_files
            ]
        }

        manifest_json = json.dumps(manifest_dict, indent=2, sort_keys=True)
        compressed_bytes = len(manifest_json.encode('utf-8'))

        ratio = total_bytes / compressed_bytes if compressed_bytes > 0 else 0

        return total_bytes, compressed_bytes, ratio

    def compress(self) -> CompressionManifest:
        """Generate compressed canon manifest"""
        print("Scanning LAW/CANON...")
        canon_files = self.scan_canon()

        print(f"Found {len(canon_files)} canon files")

        total_bytes, compressed_bytes, ratio = self.compute_compression_ratio(canon_files)

        print(f"Total size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
        print(f"Compressed: {compressed_bytes:,} bytes ({compressed_bytes / 1024:.1f} KB)")
        print(f"Compression ratio: {ratio:.1f}x")

        manifest = CompressionManifest(
            version="1.0.0",
            canon_root="LAW/CANON",
            total_files=len(canon_files),
            total_bytes=total_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
            files=[asdict(f) for f in canon_files]
        )

        # Write manifest
        output_path = self.output_dir / "canon_compressed_manifest.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(manifest), f, indent=2, sort_keys=True)

        print(f"\nCompressed manifest written to: {output_path}")

        # Also write symbol lookup table
        symbol_table = {
            f.symbol: {
                "path": f"LAW/CANON/{f.path}",
                "sha256": f.sha256,
                "size": f.size_bytes
            }
            for f in canon_files
        }

        symbol_path = self.output_dir / "canon_symbol_table.json"
        with open(symbol_path, 'w', encoding='utf-8') as f:
            json.dump(symbol_table, f, indent=2, sort_keys=True)

        print(f"Symbol table written to: {symbol_path}")

        return manifest

    def stats(self):
        """Show compression statistics"""
        manifest_path = self.output_dir / "canon_compressed_manifest.json"

        if not manifest_path.exists():
            print("No compressed manifest found. Run --compress first.")
            return

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)

        print("=== Canon Compression Statistics ===\n")
        print(f"Canon Root: {manifest_data['canon_root']}")
        print(f"Total Files: {manifest_data['total_files']}")
        print(f"Original Size: {manifest_data['total_bytes']:,} bytes ({manifest_data['total_bytes'] / 1024:.1f} KB)")
        print(f"Compressed Size: {manifest_data['compressed_bytes']:,} bytes ({manifest_data['compressed_bytes'] / 1024:.1f} KB)")
        print(f"Compression Ratio: {manifest_data['compression_ratio']:.1f}x")
        print(f"\nH(X|S) ≈ 0 when symbols point to shared canon context")
        print(f"Information transmitted: {manifest_data['compressed_bytes']:,} bytes (symbols + metadata)")
        print(f"Information referenced: {manifest_data['total_bytes']:,} bytes (full canon content)")

        print("\n=== Symbol Table ===")
        print(f"Total symbols: {manifest_data['total_files']}")
        print("Sample symbols:")
        for i, file_data in enumerate(manifest_data['files'][:5]):
            print(f"  {file_data['symbol']} → {file_data['path']} ({file_data['size']} bytes)")
        print(f"  ... and {manifest_data['total_files'] - 5} more")

    def resolve_symbol(self, symbol: str) -> str:
        """Resolve a symbol to its file path"""
        symbol_path = self.output_dir / "canon_symbol_table.json"

        if not symbol_path.exists():
            raise FileNotFoundError("Symbol table not found. Run --compress first.")

        with open(symbol_path, 'r', encoding='utf-8') as f:
            symbol_table = json.load(f)

        if symbol not in symbol_table:
            raise KeyError(f"Symbol {symbol} not found in table")

        return symbol_table[symbol]['path']


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Canon Compressor - H(X|S) symbol compression")
    parser.add_argument('--compress', action='store_true', help="Compress canon to symbols")
    parser.add_argument('--stats', action='store_true', help="Show compression statistics")
    parser.add_argument('--resolve', type=str, help="Resolve symbol to file path")

    args = parser.parse_args()

    compressor = CanonCompressor()

    if args.compress:
        compressor.compress()
    elif args.stats:
        compressor.stats()
    elif args.resolve:
        path = compressor.resolve_symbol(args.resolve)
        print(f"{args.resolve} → {path}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
