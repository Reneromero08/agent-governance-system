#!/usr/bin/env python3
"""
Section Extractor

Extracts sections from canonical sources (markdown, code) with deterministic boundaries.

Roadmap Phase: Phase 1 — Substrate + deterministic indexing
"""

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Section:
    """Canonical content unit extracted from source files."""
    section_id: str
    file_path: str
    heading_path: List[str]
    line_start: int
    line_end: int
    content_hash: str


class SectionExtractor:
    """Extracts sections from source files deterministically."""

    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize section extractor.

        Args:
            repo_root: Repository root path. Defaults to current working directory.
        """
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = repo_root

    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Text content

        Returns:
            SHA-256 hash as hex string
        """
        normalized = content.replace('\r\n', '\n')
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def compute_section_id(
        self,
        file_path: str,
        line_start: int,
        line_end: int,
        content_hash: str
    ) -> str:
        """Compute deterministic section_id.

        Args:
            file_path: Relative path from repo root
            line_start: Starting line number
            line_end: Ending line number
            content_hash: Content SHA-256 hash

        Returns:
            Section ID as SHA-256 hash
        """
        identifier = f"{file_path}:{line_start}:{line_end}:{content_hash}"
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

    def extract_from_markdown(self, file_path: Path) -> List[Section]:
        """Extract sections from markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            List of sections
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count lines the same way f.readlines() does - each newline terminates a line
                lines = len(content.splitlines())
        except Exception as e:
            raise ValueError(f"Failed to read {file_path}: {e}")
 
        try:
            relative_path = file_path.relative_to(self.repo_root).as_posix()
        except ValueError:
            relative_path = file_path.as_posix()
 
        content_hash = self.compute_content_hash(content)

        section = Section(
            section_id=self.compute_section_id(
                relative_path,
                0,
                lines,
                content_hash
            ),
            file_path=relative_path,
            heading_path=[],
            line_start=0,
            line_end=lines,
            content_hash=content_hash
        )

        return [section]

    def extract(self, file_path: Path) -> List[Section]:
        """Extract sections from file based on type.

        Args:
            file_path: Path to file

        Returns:
            List of sections

        Raises:
            ValueError: If file type not supported
        """
        suffix = file_path.suffix.lower()

        if suffix in ['.md', '.markdown']:
            return self.extract_from_markdown(file_path)
        elif suffix in ['.py', '.js', '.ts', '.sql', '.json', '.yaml', '.yml']:
            return self.extract_from_code_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")


def extract_sections(
    file_path: Path,
    repo_root: Optional[Path] = None
) -> List[Section]:
    """Convenience function to extract sections from a file.

    Args:
        file_path: Path to file
        repo_root: Repository root path

    Returns:
        List of sections
    """
    extractor = SectionExtractor(repo_root)
    return extractor.extract(file_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python section_extractor.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    extractor = SectionExtractor()

    try:
        sections = extractor.extract(file_path)
        for section in sections:
            print(f"\nSection: {section.section_id[:16]}...")
            print(f"  File: {section.file_path}")
            print(f"  Heading: {' > '.join(section.heading_path)}")
            print(f"  Lines: {section.line_start}-{section.line_end}")
            print(f"  Hash: {section.content_hash[:16]}...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
