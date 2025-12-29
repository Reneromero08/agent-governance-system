"""
Mechanical Codebase Indexer - Zero Token Cost

Scans entire codebase and stores in database WITHOUT loading into LLM context.
Uses AST parsing for Python metadata extraction.
"""

import ast
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime


class MechanicalIndexer:
    """Index codebase mechanically without LLM token usage."""

    def __init__(self, db_path: str = "CORTEX/codebase_full.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_schema()

    def _create_schema(self):
        """Create database schema for codebase index."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                hash TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                extension TEXT,
                size INTEGER,
                line_count INTEGER,
                created_at TEXT,
                content TEXT
            );

            CREATE TABLE IF NOT EXISTS python_metadata (
                file_hash TEXT PRIMARY KEY,
                functions TEXT,  -- JSON array of function names
                classes TEXT,    -- JSON array of class names
                imports TEXT,    -- JSON array of imports
                docstring TEXT,
                FOREIGN KEY (file_hash) REFERENCES files(hash)
            );

            CREATE TABLE IF NOT EXISTS file_stats (
                total_files INTEGER,
                total_size INTEGER,
                by_extension TEXT,  -- JSON object {".py": 50, ".md": 20}
                indexed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_extension ON files(extension);
            CREATE INDEX IF NOT EXISTS idx_path ON files(path);
        """)
        self.conn.commit()

    def index_file(self, file_path: Path) -> dict:
        """Index a single file mechanically."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return {"error": str(e)}

        # Compute hash
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Basic file info
        self.conn.execute("""
            INSERT OR REPLACE INTO files (hash, path, extension, size, line_count, created_at, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            file_hash,
            str(file_path),
            file_path.suffix,
            len(content),
            content.count('\n') + 1,
            datetime.now().isoformat(),
            content
        ))

        # Python-specific metadata
        if file_path.suffix == '.py':
            self._extract_python_metadata(file_hash, content)

        return {
            "hash": file_hash,
            "path": str(file_path),
            "size": len(content),
            "extension": file_path.suffix
        }

    def _extract_python_metadata(self, file_hash: str, content: str):
        """Extract Python metadata via AST parsing."""
        try:
            tree = ast.parse(content)

            # Extract functions
            functions = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]

            # Extract classes
            classes = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ]

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            imports = list(set(imports))  # Deduplicate

            # Extract module docstring
            docstring = ast.get_docstring(tree) or ""

            self.conn.execute("""
                INSERT OR REPLACE INTO python_metadata (file_hash, functions, classes, imports, docstring)
                VALUES (?, ?, ?, ?, ?)
            """, (
                file_hash,
                json.dumps(functions),
                json.dumps(classes),
                json.dumps(imports),
                docstring
            ))
        except SyntaxError:
            # File has syntax errors, skip metadata
            pass

    def index_directory(self, root: str = ".", patterns: list = None, exclude: list = None):
        """Index entire directory tree mechanically."""
        if patterns is None:
            patterns = ["**/*.py", "**/*.md", "**/*.json", "**/*.yaml", "**/*.yml"]

        if exclude is None:
            exclude = [".git", "node_modules", "__pycache__", ".venv", "venv"]

        root_path = Path(root)
        indexed = 0
        stats_by_ext = {}

        for pattern in patterns:
            for file_path in root_path.glob(pattern):
                # Skip excluded directories
                if any(exc in str(file_path) for exc in exclude):
                    continue

                result = self.index_file(file_path)
                if "error" not in result:
                    indexed += 1
                    ext = result['extension']
                    stats_by_ext[ext] = stats_by_ext.get(ext, 0) + 1

                    if indexed % 50 == 0:
                        print(f"  Indexed {indexed} files...")

        # Store stats
        total_size = self.conn.execute("SELECT SUM(size) FROM files").fetchone()[0] or 0
        self.conn.execute("""
            INSERT OR REPLACE INTO file_stats (total_files, total_size, by_extension, indexed_at)
            VALUES (?, ?, ?, ?)
        """, (indexed, total_size, json.dumps(stats_by_ext), datetime.now().isoformat()))

        self.conn.commit()
        return {
            "total_files": indexed,
            "total_size": total_size,
            "by_extension": stats_by_ext
        }

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        row = self.conn.execute("SELECT * FROM file_stats ORDER BY indexed_at DESC LIMIT 1").fetchone()
        if not row:
            return {"error": "No stats available"}

        return {
            "total_files": row[0],
            "total_size": row[1],
            "by_extension": json.loads(row[2]),
            "indexed_at": row[3]
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Run mechanical indexing."""
    print("=" * 60)
    print("MECHANICAL CODEBASE INDEXER")
    print("=" * 60)
    print()

    indexer = MechanicalIndexer()

    print("[1/3] Indexing entire codebase...")
    stats = indexer.index_directory()

    print()
    print("[2/3] Extracting statistics...")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total size: {stats['total_size']:,} bytes ({stats['total_size']/1024/1024:.2f} MB)")
    print(f"  By extension:")
    for ext, count in sorted(stats['by_extension'].items()):
        print(f"    {ext}: {count} files")

    print()
    print("[3/3] Database created successfully")
    print(f"  Location: {indexer.db_path}")
    print(f"  Size: {Path(indexer.db_path).stat().st_size / 1024:.1f} KB")

    # Python-specific stats
    py_count = indexer.conn.execute("SELECT COUNT(*) FROM python_metadata").fetchone()[0]
    if py_count > 0:
        total_functions = indexer.conn.execute("""
            SELECT SUM(json_array_length(functions)) FROM python_metadata
        """).fetchone()[0] or 0
        total_classes = indexer.conn.execute("""
            SELECT SUM(json_array_length(classes)) FROM python_metadata
        """).fetchone()[0] or 0

        print()
        print("Python Code Analysis:")
        print(f"  Files: {py_count}")
        print(f"  Functions: {total_functions}")
        print(f"  Classes: {total_classes}")

    indexer.close()

    print()
    print("=" * 60)
    print("✅ INDEXING COMPLETE (Zero LLM tokens used)")
    print("=" * 60)


if __name__ == "__main__":
    main()
