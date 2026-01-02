#!/usr/bin/env python3
"""
Section Indexer

Builds and manages the section index artifact with incremental rebuild support.

Roadmap Phase: Phase 1 — Substrate + deterministic indexing
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Iterator, Tuple
from dataclasses import asdict
from datetime import datetime

from .section_extractor import SectionExtractor, Section, extract_sections
from .slice_resolver import SliceResolver, SliceError, SliceResult
from .paths import get_cortex_dir, get_system1_db, get_sqlite_connection


class SectionIndexer:
    """Builds and manages section index with incremental rebuild support."""

    CANONICAL_SOURCES = [
        "LAW/CANON/*.md",
        "LAW/CONTEXT/**/*.md",
        "SKILLS/**/*.md",
        "TOOLS/**/*.md",
        "LAW/CONTRACTS/**/*.md",
        "NAVIGATION/CORTEX/db/**/*.sql",
    ]

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        substrate_mode: str = "sqlite"
    ):
        """Initialize section indexer.

        Args:
            repo_root: Repository root path. Defaults to current working directory.
            substrate_mode: "sqlite" or "jsonl"
        """
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = repo_root
        self.substrate_mode = substrate_mode
        self.extractor = SectionExtractor(repo_root)

        if substrate_mode == "sqlite":
            self.db_path = get_system1_db(repo_root)
        elif substrate_mode == "jsonl":
            cortex_dir = get_cortex_dir(repo_root)
            self.output_path = cortex_dir / "section_index.jsonl"
        else:
            raise ValueError(f"Invalid substrate_mode: {substrate_mode}")

    def get_canonical_files(self) -> List[Path]:
        """Get list of all canonical source files.

        Returns:
            List of file paths
        """
        files = []
        for pattern in self.CANONICAL_SOURCES:
            if "**" in pattern:
                parts = pattern.split("**")
                base_path = self.repo_root / parts[0]
                suffix_pattern = parts[1].lstrip("/")
                for file_path in base_path.rglob(suffix_pattern):
                    if file_path.is_file():
                        files.append(file_path)
            else:
                base_path = self.repo_root / pattern.rsplit("/", 1)[0]
                glob_pattern = pattern.rsplit("/", 1)[1]
                for file_path in base_path.glob(glob_pattern):
                    if file_path.is_file():
                        files.append(file_path)
        return sorted(files)

    def build_full_index(self) -> List[Section]:
        """Build full section index from all canonical sources.

        Returns:
            List of all sections
        """
        all_sections = []
        files = self.get_canonical_files()

        for file_path in files:
            try:
                sections = self.extractor.extract(file_path)
                all_sections.extend(sections)
            except Exception as e:
                print(f"Warning: Failed to extract from {file_path}: {e}")

        return all_sections

    def build_incremental_index(
        self,
        changed_files: Optional[Set[str]] = None
    ) -> List[Section]:
        """Build index incrementally, only re-indexing changed files.

        Args:
            changed_files: Set of relative file paths that changed. If None,
                          all files are re-indexed.

        Returns:
            List of all sections
        """
        all_sections = []
        files = self.get_canonical_files()

        for file_path in files:
            relative_path = file_path.relative_to(self.repo_root).as_posix()

            if changed_files is None or relative_path in changed_files:
                try:
                    sections = self.extractor.extract(file_path)
                    all_sections.extend(sections)
                except Exception as e:
                    print(f"Warning: Failed to extract from {file_path}: {e}")

        return all_sections

    def compute_index_hash(self, sections: List[Section]) -> str:
        """Compute hash of the entire index for determinism verification.

        Args:
            sections: List of sections

        Returns:
            SHA-256 hash as hex string
        """
        sorted_data = json.dumps(
            [asdict(s) for s in sorted(sections, key=lambda x: x.section_id)],
            sort_keys=True
        )
        return hashlib.sha256(sorted_data.encode('utf-8')).hexdigest()

    def write_jsonl(self, sections: List[Section]) -> None:
        """Write sections to JSONL file.

        Args:
            sections: List of sections
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_sections = sorted(sections, key=lambda x: x.section_id)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            for section in sorted_sections:
                f.write(json.dumps(asdict(section)) + '\n')

        print(f"Wrote {len(sections)} sections to {self.output_path}")

    def write_sqlite(self, sections: List[Section]) -> None:
        """Write sections to SQLite database.

        Args:
            sections: List of sections
        """
        with get_sqlite_connection(self.db_path) as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sections (
                    section_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_file ON sections(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_hash ON sections(content_hash)")

            conn.execute("DELETE FROM sections")

            for section in sections:
                conn.execute(
                    """
                    INSERT INTO sections (
                        section_id, file_path, heading_path,
                        line_start, line_end, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        section.section_id,
                        section.file_path,
                        json.dumps(section.heading_path),
                        section.line_start,
                        section.line_end,
                        section.content_hash
                    )
                )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS section_index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            index_hash = self.compute_index_hash(sections)
            timestamp = datetime.utcnow().isoformat() + "Z"
            section_count = len(sections)

            conn.execute("""
                INSERT OR REPLACE INTO section_index_meta (key, value)
                VALUES ('index_hash', ?)
            """, (index_hash,))

            conn.execute("""
                INSERT OR REPLACE INTO section_index_meta (key, value)
                VALUES ('section_count', ?)
            """, (str(section_count),))

            conn.execute("""
                INSERT OR REPLACE INTO section_index_meta (key, value)
                VALUES ('updated_at', ?)
            """, (timestamp,))

        print(f"Wrote {len(sections)} sections to {self.db_path}")

    def build(self, incremental: bool = False, changed_files: Optional[Set[str]] = None) -> str:
        """Build section index.

        Args:
            incremental: Whether to build incrementally
            changed_files: Set of changed file paths (relative)

        Returns:
            Index hash
        """
        if incremental:
            sections = self.build_incremental_index(changed_files)
        else:
            sections = self.build_full_index()

        if self.substrate_mode == "sqlite":
            self.write_sqlite(sections)
        else:
            self.write_jsonl(sections)

        index_hash = self.compute_index_hash(sections)
        print(f"Index hash: {index_hash[:16]}...")

        return index_hash

    def get_section_by_id(self, section_id: str) -> Optional[Section]:
        """Retrieve section by ID.

        Args:
            section_id: Section ID

        Returns:
            Section object or None if not found
        """
        if self.substrate_mode == "sqlite":
            with get_sqlite_connection(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM sections WHERE section_id = ?",
                    (section_id,)
                )
                row = cursor.fetchone()

                if row:
                    return Section(
                        section_id=row['section_id'],
                        file_path=row['file_path'],
                        heading_path=json.loads(row['heading_path']),
                        line_start=row['line_start'],
                        line_end=row['line_end'],
                        content_hash=row['content_hash']
                    )
                return None

        else:
            if not self.output_path.exists():
                return None

            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data['section_id'] == section_id:
                        return Section(**data)
            return None

    def read_section_content(self, section: Section) -> str:
        """Read actual content from file for a section.

        Args:
            section: Section object

        Returns:
            Section content

        Raises:
            FileNotFoundError: If file not found
        """
        file_path = self.repo_root / section.file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_idx = section.line_start
        end_idx = section.line_end

        if start_idx < 0 or end_idx > len(lines):
            raise ValueError(
                f"Invalid line range: {start_idx}-{end_idx} (file has {len(lines)} lines)"
            )

        content = ''.join(lines[start_idx:end_idx])
        normalized = content.replace('\r\n', '\n')

        actual_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        if actual_hash != section.content_hash:
            raise ValueError(f"Content hash mismatch for section {section.section_id}")

        return content

    def get_section_content(
        self,
        section_id: str,
        slice_expr: Optional[str] = None
    ) -> Tuple[str, str, str, int, int]:
        """Get section content, optionally sliced.

        Args:
            section_id: Section ID
            slice_expr: Optional slice expression (e.g., "lines[0:100]")

        Returns:
            Tuple of (content, content_hash, slice_expr, lines_applied, chars_applied)

        Raises:
            ValueError: If section not found
            SliceError: If slice expression is invalid
        """
        section = self.get_section_by_id(section_id)

        if section is None:
            raise ValueError(f"Section not found: {section_id}")

        full_content = self.read_section_content(section)
        full_content = full_content.replace('\r\n', '\n')

        if slice_expr is None:
            result_hash = hashlib.sha256(full_content.encode('utf-8')).hexdigest()
            lines_applied = full_content.count('\n') + 1
            chars_applied = len(full_content)
            return (full_content, section.content_hash, "none", lines_applied, chars_applied)

        resolver = SliceResolver()
        result = resolver.apply_slice(full_content, slice_expr)

        return (
            result.content,
            result.content_hash,
            result.slice_expr,
            result.lines_applied,
            result.chars_applied
        )

    def verify_determinism(self) -> bool:
        """Verify that two consecutive builds produce identical index.

        Returns:
            True if deterministic, False otherwise
        """
        print("First build...")
        hash1 = self.build()

        print("Second build...")
        hash2 = self.build()

        if hash1 == hash2:
            print("[OK] Index is deterministic (hashes match)")
            return True
        else:
            print(f"[FAIL] Index not deterministic (hash1={hash1[:16]}..., hash2={hash2[:16]}...)")
            return False


def build_index(
    repo_root: Optional[Path] = None,
    substrate_mode: str = "sqlite",
    incremental: bool = False
) -> str:
    """Convenience function to build section index.

    Args:
        repo_root: Repository root path
        substrate_mode: "sqlite" or "jsonl"
        incremental: Whether to build incrementally

    Returns:
        Index hash
    """
    indexer = SectionIndexer(repo_root, substrate_mode)
    return indexer.build(incremental)


if __name__ == '__main__':
    import sys

    indexer = SectionIndexer()

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        success = indexer.verify_determinism()
        sys.exit(0 if success else 1)
    else:
        indexer.build()
