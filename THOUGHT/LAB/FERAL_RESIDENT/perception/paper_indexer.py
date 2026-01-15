#!/usr/bin/env python3
"""
Paper Indexer for Feral Resident Beta (B.1)

Indexes research papers into the geometric memory for self-education.

Flow:
1. PDF -> Markdown (via /pdf-to-markdown skill)
2. Markdown -> Chunks (by ## heading structure)
3. Chunks -> GeometricStates (via reasoner.initialize)
4. Store with @Paper-{arxiv_id} symbol + @Paper-{ShortName} alias

Usage:
    indexer = PaperIndexer()
    indexer.register_paper("2310.06816", "Vec2Text", "raw/vec2text.pdf", "vec2text")
    indexer.index_paper("2310.06816")
    results = indexer.query_papers("embedding inversion", k=5)
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add parent path for imports
FERAL_PATH = Path(__file__).parent
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

# Add CAPABILITY path for geometric_reasoner
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

try:
    from geometric_reasoner import GeometricReasoner, GeometricState
except ImportError:
    GeometricReasoner = None
    GeometricState = None


@dataclass
class PaperChunk:
    """A chunk of a paper, indexed as GeometricState."""
    heading: str
    level: int
    content_hash: str
    vector_hash: str
    Df: float
    char_count: int


@dataclass
class PaperRecord:
    """A registered/indexed paper."""
    arxiv_id: str
    short_name: str
    title: str
    category: str
    primary_symbol: str
    alias_symbol: str
    pdf_path: Optional[str]
    markdown_path: Optional[str]
    status: str  # registered, converted, indexed
    chunks: List[Dict] = field(default_factory=list)
    Df_values: List[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    indexed_at: Optional[str] = None


class PaperIndexer:
    """
    Index papers into cassettes using geometric initialization.

    Hybrid symbol naming:
    - @Paper-{arxiv_id} (primary, canonical)
    - @Paper-{ShortName} (alias, human-friendly)

    Both resolve to the same GeometricState.
    """

    def __init__(self, papers_dir: Optional[str] = None):
        """
        Initialize paper indexer.

        Args:
            papers_dir: Path to papers directory. Defaults to research/papers/
        """
        if papers_dir:
            self.papers_dir = Path(papers_dir)
        else:
            self.papers_dir = FERAL_PATH / "research" / "papers"

        self.manifest_path = self.papers_dir / "manifest.json"
        self._load_manifest()

        # Initialize geometric reasoner (lazy load for performance)
        self._reasoner = None

    @property
    def reasoner(self) -> GeometricReasoner:
        """Lazy-load the geometric reasoner."""
        if self._reasoner is None:
            if GeometricReasoner is None:
                raise ImportError(
                    "GeometricReasoner not available. "
                    "Install sentence-transformers: pip install sentence-transformers"
                )
            self._reasoner = GeometricReasoner()
        return self._reasoner

    def _load_manifest(self):
        """Load or create paper manifest."""
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        else:
            self.manifest = {
                "version": "1.0.0",
                "created": datetime.utcnow().isoformat(),
                "papers": {},
                "aliases": {},
                "categories": {},
                "stats": {"total": 0, "indexed": 0}
            }
            self._save_manifest()

    def _save_manifest(self):
        """Save manifest to disk with canonical JSON formatting."""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding='utf-8'
        )

    def register_paper(
        self,
        arxiv_id: str,
        short_name: str,
        title: str,
        category: str,
        pdf_path: Optional[str] = None
    ) -> PaperRecord:
        """
        Register a paper for indexing.

        Creates:
        - @Paper-{arxiv_id} (primary symbol)
        - @Paper-{short_name} (alias)

        Args:
            arxiv_id: Arxiv paper ID (e.g., "2310.06816")
            short_name: Human-friendly name (e.g., "Vec2Text")
            title: Full paper title
            category: Category (vec2text, hdc_vsa, latent_reasoning, etc.)
            pdf_path: Path to PDF file (relative to papers/raw/)

        Returns:
            PaperRecord with registration details
        """
        paper = PaperRecord(
            arxiv_id=arxiv_id,
            short_name=short_name,
            title=title,
            category=category,
            primary_symbol=f"@Paper-{arxiv_id}",
            alias_symbol=f"@Paper-{short_name}",
            pdf_path=pdf_path,
            markdown_path=None,
            status="registered"
        )

        self.manifest["papers"][arxiv_id] = asdict(paper)
        self.manifest["aliases"][short_name] = arxiv_id
        self.manifest["stats"]["total"] += 1
        self._save_manifest()

        return paper

    def get_paper(self, identifier: str) -> Optional[Dict]:
        """
        Get paper by arxiv_id or short_name.

        Args:
            identifier: Either arxiv_id or short_name

        Returns:
            Paper dict or None if not found
        """
        # Check if it's an alias
        if identifier in self.manifest["aliases"]:
            arxiv_id = self.manifest["aliases"][identifier]
        else:
            arxiv_id = identifier

        return self.manifest["papers"].get(arxiv_id)

    def set_markdown_path(self, arxiv_id: str, markdown_path: str) -> Dict:
        """
        Set the markdown path for a paper after PDF conversion.

        Args:
            arxiv_id: Paper arxiv ID
            markdown_path: Path to converted markdown file

        Returns:
            Updated paper dict
        """
        if arxiv_id not in self.manifest["papers"]:
            raise ValueError(f"Paper {arxiv_id} not registered")

        paper = self.manifest["papers"][arxiv_id]
        paper["markdown_path"] = markdown_path
        paper["status"] = "converted"
        self._save_manifest()

        return paper

    def chunk_by_headings(self, markdown_path: str) -> List[Dict]:
        """
        Chunk markdown by heading structure (# ## ###).

        Returns list of:
        {
            "heading": "## Introduction",
            "level": 2,
            "content": "...",
            "hash": "sha256:..."
        }
        """
        text = Path(markdown_path).read_text(encoding='utf-8')
        chunks = []
        current_chunk = {"heading": "# Preamble", "level": 0, "content": ""}

        for line in text.split('\n'):
            # Check for markdown headings
            if line.startswith('#') and not line.startswith('#!'):
                # Count heading level
                stripped = line.lstrip('#')
                level = len(line) - len(stripped)

                # Only treat as heading if there's a space after #s
                if stripped.startswith(' ') or stripped == '':
                    # Save previous chunk if non-empty
                    if current_chunk["content"].strip():
                        content_bytes = current_chunk["content"].encode('utf-8')
                        current_chunk["hash"] = hashlib.sha256(content_bytes).hexdigest()[:16]
                        current_chunk["char_count"] = len(current_chunk["content"])
                        chunks.append(current_chunk)

                    # Start new chunk
                    current_chunk = {
                        "heading": line.strip(),
                        "level": level,
                        "content": ""
                    }
                    continue

            current_chunk["content"] += line + "\n"

        # Don't forget last chunk
        if current_chunk["content"].strip():
            content_bytes = current_chunk["content"].encode('utf-8')
            current_chunk["hash"] = hashlib.sha256(content_bytes).hexdigest()[:16]
            current_chunk["char_count"] = len(current_chunk["content"])
            chunks.append(current_chunk)

        return chunks

    def index_paper(self, arxiv_id: str, max_chunk_chars: int = 2000) -> Dict:
        """
        Full indexing pipeline for a paper.

        1. Check markdown exists (must be converted first via /pdf-to-markdown)
        2. Chunk by headings
        3. Initialize each chunk to GeometricState
        4. Track Df values
        5. Store with receipts

        Args:
            arxiv_id: Paper arxiv ID
            max_chunk_chars: Maximum characters per chunk for embedding

        Returns:
            Updated paper dict with chunks and Df values
        """
        if arxiv_id not in self.manifest["papers"]:
            raise ValueError(f"Paper {arxiv_id} not registered")

        paper = self.manifest["papers"][arxiv_id]

        if not paper.get("markdown_path"):
            raise ValueError(
                f"Paper {arxiv_id} has no markdown_path. "
                "Convert PDF first using /pdf-to-markdown skill."
            )

        markdown_path = Path(paper["markdown_path"])
        if not markdown_path.is_absolute():
            markdown_path = self.papers_dir / paper["markdown_path"]

        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

        # Chunk by headings
        chunks = self.chunk_by_headings(str(markdown_path))

        # Index each chunk
        indexed_chunks = []
        for chunk in chunks:
            # Truncate long chunks for embedding
            content = chunk["content"][:max_chunk_chars]

            # Initialize to manifold (boundary operation)
            state = self.reasoner.initialize(content)

            indexed_chunk = PaperChunk(
                heading=chunk["heading"],
                level=chunk["level"],
                content_hash=chunk["hash"],
                vector_hash=state.receipt()["vector_hash"],
                Df=state.Df,
                char_count=chunk.get("char_count", len(content))
            )
            indexed_chunks.append(asdict(indexed_chunk))

        paper["chunks"] = indexed_chunks
        paper["Df_values"] = [c["Df"] for c in indexed_chunks]
        paper["status"] = "indexed"
        paper["indexed_at"] = datetime.utcnow().isoformat()

        self.manifest["stats"]["indexed"] += 1
        self._save_manifest()

        return paper

    def query_papers(self, query: str, k: int = 10) -> List[Dict]:
        """
        Query papers using E (Born rule) for relevance.

        Returns top-k paper chunks by E value.

        Note: This is a simplified implementation that re-embeds chunks.
        For production, store vectors in SQLite and load for comparison.
        """
        query_state = self.reasoner.initialize(query)

        results = []
        for arxiv_id, paper in self.manifest["papers"].items():
            if paper["status"] != "indexed":
                continue

            # Load markdown and re-chunk for embedding comparison
            # (In production, load stored vectors from DB)
            markdown_path = paper.get("markdown_path")
            if not markdown_path:
                continue

            full_path = Path(markdown_path)
            if not full_path.is_absolute():
                full_path = self.papers_dir / markdown_path

            if not full_path.exists():
                continue

            chunks = self.chunk_by_headings(str(full_path))

            for chunk, stored_chunk in zip(chunks, paper.get("chunks", [])):
                # Re-embed chunk content
                content = chunk["content"][:2000]
                chunk_state = self.reasoner.initialize(content)

                # Compute E (Born rule)
                E = query_state.E_with(chunk_state)

                results.append({
                    "paper": paper["primary_symbol"],
                    "alias": paper["alias_symbol"],
                    "title": paper.get("title", "Unknown"),
                    "heading": chunk["heading"],
                    "E": E,
                    "Df": stored_chunk.get("Df", chunk_state.Df),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })

        # Sort by E (Born rule) - highest first
        results.sort(key=lambda x: x["E"], reverse=True)
        return results[:k]

    def list_papers(self, category: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """
        List registered papers.

        Args:
            category: Filter by category
            status: Filter by status (registered, converted, indexed)

        Returns:
            List of paper dicts
        """
        papers = []
        for arxiv_id, paper in self.manifest["papers"].items():
            if category and paper.get("category") != category:
                continue
            if status and paper.get("status") != status:
                continue
            papers.append(paper)

        return papers

    def get_stats(self) -> Dict:
        """Get paper corpus statistics."""
        stats = self.manifest["stats"].copy()

        # Count by status
        status_counts = {"registered": 0, "converted": 0, "indexed": 0}
        for paper in self.manifest["papers"].values():
            status = paper.get("status", "registered")
            status_counts[status] = status_counts.get(status, 0) + 1

        stats["by_status"] = status_counts

        # Count by category
        category_counts = {}
        for paper in self.manifest["papers"].values():
            cat = paper.get("category", "uncategorized")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        stats["by_category"] = category_counts

        # Compute Df statistics across indexed papers
        all_Df = []
        for paper in self.manifest["papers"].values():
            all_Df.extend(paper.get("Df_values", []))

        if all_Df:
            stats["Df_min"] = min(all_Df)
            stats["Df_max"] = max(all_Df)
            stats["Df_mean"] = sum(all_Df) / len(all_Df)
        else:
            stats["Df_min"] = stats["Df_max"] = stats["Df_mean"] = 0.0

        return stats


# === Utility Functions ===

def index_markdown_directly(markdown_path: str, arxiv_id: str, short_name: str,
                            title: str, category: str) -> Dict:
    """
    Convenience function to index a markdown file directly.

    Skips PDF conversion step - use when you already have markdown.
    """
    indexer = PaperIndexer()

    # Register
    indexer.register_paper(arxiv_id, short_name, title, category)

    # Set markdown path
    indexer.set_markdown_path(arxiv_id, markdown_path)

    # Index
    return indexer.index_paper(arxiv_id)


if __name__ == "__main__":
    # Quick test
    indexer = PaperIndexer()
    print(f"Paper corpus: {indexer.papers_dir}")
    print(f"Stats: {indexer.get_stats()}")
