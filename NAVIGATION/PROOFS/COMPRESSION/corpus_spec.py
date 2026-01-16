#!/usr/bin/env python3
"""
Phase 6.4.6-6.4.7: Corpus and Context Specifications

Defines:
- Baseline corpus: explicit file allowlist with integrity anchors (6.4.6)
- Compressed context: retrieval method, parameters, deterministic tie-breaking (6.4.7)

All specifications must be reproducible and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[3]

# Corpus anchor files
FILE_INDEX_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "meta" / "FILE_INDEX.json"
SECTION_INDEX_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "meta" / "SECTION_INDEX.json"


def _sha256_hex(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    content = path.read_text(encoding="utf-8")
    return _sha256_hex(content)


@dataclass
class CorpusAnchor:
    """
    Integrity anchor for baseline corpus (6.4.6).

    Provides deterministic verification that corpus hasn't changed.
    """
    file_index_path: str
    file_index_sha256: str
    section_index_path: str
    section_index_sha256: str
    git_rev: Optional[str] = None
    timestamp_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_index_path": self.file_index_path,
            "file_index_sha256": self.file_index_sha256,
            "section_index_path": self.section_index_path,
            "section_index_sha256": self.section_index_sha256,
            "git_rev": self.git_rev,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def compute(cls, git_rev: Optional[str] = None) -> "CorpusAnchor":
        """Compute current corpus anchor from indices."""
        import datetime

        if not FILE_INDEX_PATH.exists():
            raise FileNotFoundError(f"FILE_INDEX not found: {FILE_INDEX_PATH}")
        if not SECTION_INDEX_PATH.exists():
            raise FileNotFoundError(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}")

        return cls(
            file_index_path=str(FILE_INDEX_PATH.relative_to(REPO_ROOT)),
            file_index_sha256=_sha256_file(FILE_INDEX_PATH),
            section_index_path=str(SECTION_INDEX_PATH.relative_to(REPO_ROOT)),
            section_index_sha256=_sha256_file(SECTION_INDEX_PATH),
            git_rev=git_rev,
            timestamp_utc=datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        )

    def verify(self) -> bool:
        """Verify corpus anchor matches current state."""
        try:
            current = CorpusAnchor.compute()
            return (
                self.file_index_sha256 == current.file_index_sha256 and
                self.section_index_sha256 == current.section_index_sha256
            )
        except FileNotFoundError:
            return False


@dataclass
class BaselineCorpusSpec:
    """
    Baseline corpus specification (6.4.6).

    Defines what files are included and how token counts are aggregated.
    """
    # Aggregation mode
    aggregation_mode: str = "sum_per_file"  # or "tokenize_concatenated"

    # Path filters (explicit allowlist)
    include_patterns: List[str] = field(default_factory=lambda: [
        "LAW/**/*.md",
        "LAW/**/*.json",
        "NAVIGATION/**/*.md",
        "NAVIGATION/**/*.py",
        "CANON/**/*.md",
        "CAPABILITY/**/*.py",
        "ADR-*.md",
        "**/decisions/**/*.md",
        "**/*ROADMAP*.md",
    ])

    # Explicit excludes
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/_test_*/**",
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/*.pyc",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation_mode": self.aggregation_mode,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
        }

    def get_file_allowlist(self) -> List[str]:
        """
        Get explicit file allowlist from FILE_INDEX.json.

        Returns sorted list of file paths for deterministic ordering.
        """
        if not FILE_INDEX_PATH.exists():
            return []

        file_index = json.loads(FILE_INDEX_PATH.read_text(encoding="utf-8"))
        return sorted(file_index.keys())

    def get_filtered_files(self) -> List[str]:
        """
        Get files matching include/exclude patterns.

        Uses FILE_INDEX.json as the source of truth.
        """
        import fnmatch

        all_files = self.get_file_allowlist()
        filtered = []

        for file_path in all_files:
            # Check include patterns
            included = False
            for pattern in self.include_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    included = True
                    break

            # Check exclude patterns
            if included:
                for pattern in self.exclude_patterns:
                    if fnmatch.fnmatch(file_path, pattern):
                        included = False
                        break

            if included:
                filtered.append(file_path)

        return sorted(filtered)


@dataclass
class CompressedContextSpec:
    """
    Compressed context specification (6.4.7).

    Defines retrieval method, parameters, and deterministic tie-breaking.
    """
    # Retrieval method
    retrieval_method: str = "semantic"  # or "fts_fallback"

    # Semantic search parameters
    top_k: int = 10
    min_similarity: float = 0.4
    similarity_threshold_steps: List[float] = field(default_factory=lambda: [0.50, 0.40, 0.00])

    # FTS fallback parameters
    fts_enabled: bool = True
    fts_top_k: int = 5

    # Deterministic tie-breaking (6.4.7 requirement)
    # Results sorted by: (similarity DESC, hash ASC)
    tie_break_order: str = "similarity_desc_hash_asc"

    # Retrieved identifiers are recorded as hashes
    record_retrieved_hashes: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_method": self.retrieval_method,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "similarity_threshold_steps": self.similarity_threshold_steps,
            "fts_enabled": self.fts_enabled,
            "fts_top_k": self.fts_top_k,
            "tie_break_order": self.tie_break_order,
            "record_retrieved_hashes": self.record_retrieved_hashes,
        }

    def apply_tie_breaking(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply deterministic tie-breaking to results.

        Ensures reproducible ordering regardless of underlying search implementation.
        """
        if self.tie_break_order == "similarity_desc_hash_asc":
            return sorted(
                results,
                key=lambda r: (-r.get("similarity", 0), r.get("hash", "")),
            )
        else:
            # Default: just sort by hash for determinism
            return sorted(results, key=lambda r: r.get("hash", ""))


@dataclass
class ProofCorpusSpec:
    """
    Combined corpus specification for proofs.

    Bundles baseline spec, compressed spec, and anchor for reproducibility.
    """
    spec_version: str = "1.0.0"
    baseline: BaselineCorpusSpec = field(default_factory=BaselineCorpusSpec)
    compressed: CompressedContextSpec = field(default_factory=CompressedContextSpec)
    anchor: Optional[CorpusAnchor] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "baseline": self.baseline.to_dict(),
            "compressed": self.compressed.to_dict(),
            "anchor": self.anchor.to_dict() if self.anchor else None,
        }

    def compute_anchor(self, git_rev: Optional[str] = None) -> None:
        """Compute and attach corpus anchor."""
        self.anchor = CorpusAnchor.compute(git_rev=git_rev)

    def verify_anchor(self) -> bool:
        """Verify corpus hasn't changed since anchor was computed."""
        if self.anchor is None:
            return False
        return self.anchor.verify()


def get_default_spec() -> ProofCorpusSpec:
    """Get default proof corpus specification with computed anchor."""
    spec = ProofCorpusSpec()
    try:
        spec.compute_anchor()
    except FileNotFoundError:
        pass  # Anchor will be None if indices don't exist
    return spec


def render_spec_report(spec: ProofCorpusSpec) -> str:
    """Render human-readable specification report."""
    lines = [
        "# Corpus Specification Report",
        "",
        f"**Spec Version:** {spec.spec_version}",
        "",
        "## Baseline Corpus (6.4.6)",
        "",
        f"**Aggregation Mode:** {spec.baseline.aggregation_mode}",
        "",
        "### Include Patterns",
        "",
    ]

    for pattern in spec.baseline.include_patterns:
        lines.append(f"- `{pattern}`")

    lines.extend([
        "",
        "### Exclude Patterns",
        "",
    ])

    for pattern in spec.baseline.exclude_patterns:
        lines.append(f"- `{pattern}`")

    lines.extend([
        "",
        "## Compressed Context (6.4.7)",
        "",
        f"**Retrieval Method:** {spec.compressed.retrieval_method}",
        f"**Top K:** {spec.compressed.top_k}",
        f"**Min Similarity:** {spec.compressed.min_similarity}",
        f"**Tie-Breaking:** {spec.compressed.tie_break_order}",
        f"**FTS Fallback:** {'Enabled' if spec.compressed.fts_enabled else 'Disabled'}",
        "",
        "## Corpus Anchor",
        "",
    ])

    if spec.anchor:
        lines.extend([
            f"**FILE_INDEX:** `{spec.anchor.file_index_path}`",
            f"**FILE_INDEX SHA256:** `{spec.anchor.file_index_sha256[:16]}...`",
            f"**SECTION_INDEX:** `{spec.anchor.section_index_path}`",
            f"**SECTION_INDEX SHA256:** `{spec.anchor.section_index_sha256[:16]}...`",
            f"**Git Rev:** `{spec.anchor.git_rev or 'unknown'}`",
            f"**Timestamp:** {spec.anchor.timestamp_utc or 'unknown'}",
        ])
    else:
        lines.append("*Anchor not computed (indices may not exist)*")

    lines.extend([
        "",
        "---",
        "",
        "*Phase 6.4.6-6.4.7 compliant specification.*",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Generate and print spec
    spec = get_default_spec()
    print(json.dumps(spec.to_dict(), indent=2))
    print("\n" + "=" * 60 + "\n")
    print(render_spec_report(spec))
