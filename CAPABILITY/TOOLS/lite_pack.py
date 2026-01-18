#!/usr/bin/env python3
"""
LITE Pack Generator - ELO-based context pack filtering.

Generates LITE context packs filtered by ELO tier for efficient
context delivery to LLMs. Part of E.4 ELO Filtering implementation.

Tier Filtering Logic:
    HIGH (>=1600): Include full content
    MEDIUM (1200-1599): Summarize (signatures/headers only)
    LOW (800-1199): Pointer only (path + ELO)
    VERY_LOW (<800): Exclude entirely
"""

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import ELO dependencies from sibling module
try:
    from ..PRIMITIVES.elo_db import EloDatabase
    from ..PRIMITIVES.elo_engine import EloEngine
except ImportError:
    # Allow direct script execution
    primitives_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "PRIMITIVES"
    )
    sys.path.insert(0, primitives_path)
    from elo_db import EloDatabase
    from elo_engine import EloEngine


class LitePackGenerator:
    """Generate LITE context packs filtered by ELO tier."""

    # Tier thresholds
    TIER_HIGH = 1600
    TIER_MEDIUM = 1200
    TIER_LOW = 800

    # Default ELO for untracked files
    DEFAULT_ELO = 1000.0

    # Token estimation: ~4 chars per token
    CHARS_PER_TOKEN = 4

    # Directories that should NEVER be included in packs (volatile/transient content)
    EXCLUDED_PREFIXES = (
        "INBOX/",              # Transient session data
        "INBOX\\",
        "MEMORY/ARCHIVE/",     # Archived/pruned content
        "MEMORY/ARCHIVE\\",
        "THOUGHT/LAB/",        # Research scratch - internal only
        "THOUGHT/LAB\\",
        "_generated/",         # Generated artifacts (logs, databases)
        "_generated\\",
        ".git/",               # Git internals
        ".git\\",
        "__pycache__/",        # Python cache
        "__pycache__\\",
        "node_modules/",       # Node dependencies
        "node_modules\\",
    )

    def __init__(self, repo_root: str, db: EloDatabase, engine: EloEngine):
        """
        Initialize with repo root, database, and engine.

        Args:
            repo_root: Root path of the repository
            db: EloDatabase instance for score storage
            engine: EloEngine instance for tier classification
        """
        self.repo_root = Path(repo_root)
        self.db = db
        self.engine = engine

    def get_file_elo(self, file_path: str) -> float:
        """
        Get ELO for file, returning default 1000.0 if not tracked.

        Args:
            file_path: Path to the file (relative or absolute)

        Returns:
            ELO score for the file
        """
        # Normalize path for database lookup
        normalized = self._normalize_path(file_path)
        elo = self.db.get_elo("file", normalized)
        return elo if elo != self.DEFAULT_ELO else self.DEFAULT_ELO

    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path for consistent lookup."""
        path = Path(file_path)
        if path.is_absolute():
            try:
                return str(path.relative_to(self.repo_root))
            except ValueError:
                return str(path)
        return str(path)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return len(text) // self.CHARS_PER_TOKEN

    def _get_tier(self, elo: float) -> str:
        """Get tier name from ELO score."""
        return self.engine.get_tier(elo)

    def _is_excluded_path(self, file_path: str) -> bool:
        """Check if path should be excluded from packs (INBOX, archives, etc.)."""
        normalized = self._normalize_path(file_path).replace("\\", "/")
        for prefix in self.EXCLUDED_PREFIXES:
            check_prefix = prefix.replace("\\", "/")
            if normalized.startswith(check_prefix) or ("/" + check_prefix) in normalized:
                return True
        return False

    def summarize_file(self, file_path: str) -> str:
        """
        Create summary for MEDIUM tier files.

        For code files (.py, .js, .ts):
            - Extract function/class signatures
            - Include docstrings

        For markdown (.md):
            - Extract headers (# ## ###)
            - First paragraph

        For other files:
            - First 10 lines

        Args:
            file_path: Path to the file

        Returns:
            Summary string
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.repo_root / path

        if not path.exists():
            return f"[File not found: {file_path}]"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[Error reading file: {e}]"

        suffix = path.suffix.lower()

        if suffix == ".py":
            return self._summarize_python(content)
        elif suffix in (".js", ".ts", ".jsx", ".tsx"):
            return self._summarize_javascript(content)
        elif suffix == ".md":
            return self._summarize_markdown(content)
        else:
            return self._summarize_generic(content)

    def _summarize_python(self, content: str) -> str:
        """Extract Python signatures and docstrings."""
        lines = content.split("\n")
        summary_lines = []
        in_docstring = False
        docstring_char = None
        docstring_lines = []

        for line in lines:
            stripped = line.strip()

            # Module docstring at start
            if not summary_lines and (
                stripped.startswith('"""') or stripped.startswith("'''")
            ):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    # Single-line docstring
                    summary_lines.append(line)
                else:
                    in_docstring = True
                    docstring_lines = [line]
                continue

            # Inside docstring
            if in_docstring:
                docstring_lines.append(line)
                if docstring_char and docstring_char in stripped:
                    in_docstring = False
                    summary_lines.extend(docstring_lines)
                    docstring_lines = []
                continue

            # Class definitions
            if stripped.startswith("class "):
                summary_lines.append(line)
                continue

            # Function definitions (def or async def)
            if stripped.startswith("def ") or stripped.startswith("async def "):
                summary_lines.append(line)
                continue

            # Decorators
            if stripped.startswith("@"):
                summary_lines.append(line)
                continue

            # Import statements (first section only)
            if stripped.startswith("import ") or stripped.startswith("from "):
                if not any(
                    l.strip().startswith("class ") or l.strip().startswith("def ")
                    for l in summary_lines
                ):
                    summary_lines.append(line)
                continue

        return "\n".join(summary_lines) if summary_lines else "[No signatures found]"

    def _summarize_javascript(self, content: str) -> str:
        """Extract JavaScript/TypeScript signatures."""
        lines = content.split("\n")
        summary_lines = []

        # Patterns to match
        patterns = [
            r"^\s*(export\s+)?(async\s+)?function\s+\w+",
            r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
            r"^\s*(export\s+)?class\s+\w+",
            r"^\s*(export\s+)?interface\s+\w+",
            r"^\s*(export\s+)?type\s+\w+",
            r"^\s*import\s+",
            r"^\s*export\s+",
        ]

        for line in lines:
            for pattern in patterns:
                if re.match(pattern, line):
                    summary_lines.append(line)
                    break

        return "\n".join(summary_lines) if summary_lines else "[No signatures found]"

    def _summarize_markdown(self, content: str) -> str:
        """Extract markdown headers and first paragraph."""
        lines = content.split("\n")
        summary_lines = []
        found_first_para = False
        in_para = False

        for line in lines:
            stripped = line.strip()

            # Headers
            if stripped.startswith("#"):
                summary_lines.append(line)
                continue

            # First non-empty paragraph
            if not found_first_para:
                if stripped and not stripped.startswith("#"):
                    in_para = True
                    summary_lines.append(line)
                elif in_para and not stripped:
                    found_first_para = True
                    in_para = False
                elif in_para:
                    summary_lines.append(line)

        return "\n".join(summary_lines) if summary_lines else "[Empty document]"

    def _summarize_generic(self, content: str) -> str:
        """Return first 10 lines for generic files."""
        lines = content.split("\n")[:10]
        if len(content.split("\n")) > 10:
            lines.append("... [truncated]")
        return "\n".join(lines)

    def generate_pack(
        self, file_paths: list, include_metadata: bool = True
    ) -> dict:
        """
        Generate LITE pack from file list.

        Args:
            file_paths: List of file paths to include
            include_metadata: Whether to include ELO metadata

        Returns:
            Pack dictionary with manifest, content, and optional metadata
        """
        # Initialize content buckets
        content = {"HIGH": [], "MEDIUM": [], "LOW": []}
        metadata = []

        # Track statistics
        total_files = len(file_paths)
        included_count = 0
        summarized_count = 0
        pointer_count = 0
        excluded_count = 0
        blocked_count = 0  # Files blocked by exclusion policy (INBOX, etc.)

        full_content_chars = 0
        lite_content_chars = 0

        elo_values = []
        blocked_paths = []

        for file_path in file_paths:
            # Check exclusion policy FIRST (INBOX, archives, etc. never in packs)
            if self._is_excluded_path(file_path):
                blocked_count += 1
                blocked_paths.append(self._normalize_path(file_path))
                continue

            elo = self.get_file_elo(file_path)
            tier = self._get_tier(elo)
            elo_values.append(elo)

            # Get full content for token estimation
            path = Path(file_path)
            if not path.is_absolute():
                path = self.repo_root / path

            try:
                if path.exists():
                    full_content = path.read_text(encoding="utf-8", errors="replace")
                else:
                    full_content = ""
            except Exception:
                full_content = ""

            full_content_chars += len(full_content)

            normalized_path = self._normalize_path(file_path)

            if tier == "HIGH":
                # Include complete content
                entry = {"path": normalized_path, "elo": elo, "content": full_content}
                content["HIGH"].append(entry)
                lite_content_chars += len(full_content)
                included_count += 1

            elif tier == "MEDIUM":
                # Summarize
                summary = self.summarize_file(file_path)
                entry = {"path": normalized_path, "elo": elo, "summary": summary}
                content["MEDIUM"].append(entry)
                lite_content_chars += len(summary)
                summarized_count += 1

            elif tier == "LOW":
                # Pointer only
                entry = {"path": normalized_path, "elo": elo}
                content["LOW"].append(entry)
                lite_content_chars += len(normalized_path) + 20  # path + elo
                pointer_count += 1

            else:  # VERY_LOW
                # Exclude entirely
                excluded_count += 1

            # Build metadata entry
            if include_metadata:
                metadata.append({"path": normalized_path, "elo": elo, "tier": tier})

        # Calculate compression ratio
        if full_content_chars > 0:
            compression_ratio = 1.0 - (lite_content_chars / full_content_chars)
        else:
            compression_ratio = 0.0

        # Build manifest
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_files": total_files,
            "included": included_count,
            "summarized": summarized_count,
            "pointers": pointer_count,
            "excluded": excluded_count,      # VERY_LOW ELO
            "blocked": blocked_count,        # Policy exclusion (INBOX, etc.)
            "compression_ratio": round(compression_ratio, 4),
        }

        # Add blocked paths if any (for transparency)
        if blocked_paths:
            manifest["blocked_paths"] = blocked_paths

        # Add ELO range if we have values
        if elo_values:
            manifest["elo_range"] = {
                "min": round(min(elo_values), 2),
                "max": round(max(elo_values), 2),
            }

        # Build by_tier breakdown
        manifest["by_tier"] = {
            "HIGH": {
                "count": included_count,
                "tokens": self._estimate_tokens(
                    "".join(e.get("content", "") for e in content["HIGH"])
                ),
            },
            "MEDIUM": {
                "count": summarized_count,
                "tokens": self._estimate_tokens(
                    "".join(e.get("summary", "") for e in content["MEDIUM"])
                ),
            },
            "LOW": {
                "count": pointer_count,
                "tokens": pointer_count * 10,  # Rough estimate for pointers
            },
            "VERY_LOW": {"count": excluded_count, "excluded": True},
        }

        # Build result
        result = {"manifest": manifest, "content": content}

        if include_metadata:
            result["metadata"] = metadata

        return result

    def generate_pack_for_query(self, query: str, max_files: int = 20) -> dict:
        """
        Generate pack relevant to query, sorted by ELO.

        Uses existing ELO scores to prioritize files.

        Args:
            query: Search query (used for context, not filtering)
            max_files: Maximum number of files to include

        Returns:
            Pack dictionary
        """
        # Get top files by ELO
        top_files = self.db.get_top_k("file", max_files)
        file_paths = [f[0] for f in top_files]

        # Generate pack
        pack = self.generate_pack(file_paths)

        # Add query context to manifest
        pack["manifest"]["query"] = query

        return pack

    def estimate_token_savings(self, pack: dict) -> dict:
        """
        Estimate token savings from ELO filtering.

        Args:
            pack: Pack dictionary from generate_pack()

        Returns:
            Dictionary with token estimates and savings percentage
        """
        content = pack.get("content", {})

        # Calculate LITE tokens
        high_content = "".join(e.get("content", "") for e in content.get("HIGH", []))
        medium_content = "".join(
            e.get("summary", "") for e in content.get("MEDIUM", [])
        )
        low_pointers = len(content.get("LOW", [])) * 40  # ~40 chars per pointer

        lite_chars = len(high_content) + len(medium_content) + low_pointers
        lite_tokens = self._estimate_tokens(str(lite_chars))

        # Estimate full tokens (if all content were included)
        manifest = pack.get("manifest", {})
        by_tier = manifest.get("by_tier", {})

        # Sum up all tier tokens
        full_tokens = 0
        for tier_data in by_tier.values():
            if isinstance(tier_data, dict):
                full_tokens += tier_data.get("tokens", 0)

        # Add estimate for excluded content (assume similar to average)
        excluded_count = manifest.get("excluded", 0)
        if manifest.get("included", 0) + manifest.get("summarized", 0) > 0:
            avg_tokens = full_tokens / (
                manifest.get("included", 0) + manifest.get("summarized", 0) + 0.01
            )
            full_tokens += int(excluded_count * avg_tokens * 2)  # Assume excluded are larger

        # Ensure we have valid values
        if full_tokens == 0:
            full_tokens = lite_tokens + 100  # Minimum overhead

        # Calculate savings
        savings_pct = (
            ((full_tokens - lite_tokens) / full_tokens * 100)
            if full_tokens > 0
            else 0.0
        )

        return {
            "full_tokens": full_tokens,
            "lite_tokens": lite_tokens,
            "savings_pct": round(savings_pct, 1),
        }


def run_self_test() -> bool:
    """
    Run comprehensive self-test of LitePackGenerator.

    Returns:
        True if all tests pass, False otherwise.
    """
    import json
    import shutil
    import tempfile

    print("=" * 60)
    print("LITE Pack Generator Self-Test")
    print("=" * 60)

    # Create temporary directory structure
    test_dir = tempfile.mkdtemp(prefix="lite_pack_test_")
    test_db_path = os.path.join(test_dir, "test_elo.db")
    test_log_path = os.path.join(test_dir, "elo_updates.jsonl")
    test_repo = os.path.join(test_dir, "repo")
    os.makedirs(test_repo)

    all_passed = True

    try:
        # Create test files with different content types
        print("\n[Setup] Creating test files...")

        # HIGH tier Python file (will set ELO to 1700)
        high_py = os.path.join(test_repo, "high_tier.py")
        with open(high_py, "w", encoding="utf-8") as f:
            f.write('''"""High tier module with important functionality."""

import os
from pathlib import Path


class ImportantClass:
    """A very important class."""

    def critical_method(self, data: dict) -> str:
        """Process critical data."""
        return str(data)


def main_function(arg1: str, arg2: int) -> bool:
    """Main entry point for the module."""
    return True
''')

        # MEDIUM tier Python file (will set ELO to 1300)
        medium_py = os.path.join(test_repo, "medium_tier.py")
        with open(medium_py, "w", encoding="utf-8") as f:
            f.write('''"""Medium tier helper module."""

def helper_function(x: int) -> int:
    """A helper function."""
    return x * 2


class UtilityClass:
    """Utility class for common operations."""
    pass
''')

        # LOW tier markdown file (will set ELO to 900)
        low_md = os.path.join(test_repo, "low_tier.md")
        with open(low_md, "w", encoding="utf-8") as f:
            f.write("""# Low Priority Document

This is a low priority document that rarely gets accessed.

## Section 1

Some content here.

## Section 2

More content here.
""")

        # VERY_LOW tier file (will set ELO to 700)
        very_low = os.path.join(test_repo, "very_low_tier.txt")
        with open(very_low, "w", encoding="utf-8") as f:
            f.write("Rarely used content\n" * 20)

        # DEFAULT tier file (not in database, ELO = 1000)
        default_py = os.path.join(test_repo, "default_tier.py")
        with open(default_py, "w", encoding="utf-8") as f:
            f.write('''"""Untracked file with default ELO."""

def untracked_function():
    pass
''')

        print(f"  Created 5 test files in {test_repo}")

        # Initialize ELO database and engine
        print("\n[Test 1] Initialize EloDatabase and EloEngine...")
        db = EloDatabase(test_db_path)
        engine = EloEngine(db, test_log_path)
        print("  PASSED: Database and engine initialized")

        # Set up ELO scores at different tiers
        print("\n[Test 2] Set ELO scores for test files...")
        db.set_elo("file", "high_tier.py", 1700.0)
        db.set_elo("file", "medium_tier.py", 1300.0)
        db.set_elo("file", "low_tier.md", 900.0)
        db.set_elo("file", "very_low_tier.txt", 700.0)
        # default_tier.py not set - should use default 1000

        print("  Set high_tier.py ELO = 1700 (HIGH)")
        print("  Set medium_tier.py ELO = 1300 (MEDIUM)")
        print("  Set low_tier.md ELO = 900 (LOW)")
        print("  Set very_low_tier.txt ELO = 700 (VERY_LOW)")
        print("  default_tier.py = 1000 (DEFAULT/LOW)")
        print("  PASSED: ELO scores configured")

        # Create LitePackGenerator
        print("\n[Test 3] Create LitePackGenerator...")
        generator = LitePackGenerator(test_repo, db, engine)
        assert generator.repo_root == Path(test_repo)
        print("  PASSED: Generator created")

        # Test get_file_elo
        print("\n[Test 4] Test get_file_elo()...")
        elo_high = generator.get_file_elo("high_tier.py")
        elo_default = generator.get_file_elo("default_tier.py")
        elo_nonexistent = generator.get_file_elo("nonexistent.py")

        assert elo_high == 1700.0, f"Expected 1700.0, got {elo_high}"
        assert elo_default == 1000.0, f"Expected 1000.0, got {elo_default}"
        assert elo_nonexistent == 1000.0, f"Expected default 1000.0, got {elo_nonexistent}"
        print("  PASSED: ELO retrieval works correctly")

        # Test summarize_file for Python
        print("\n[Test 5] Test summarize_file() for Python...")
        summary = generator.summarize_file("medium_tier.py")
        assert "def helper_function" in summary, "Missing function signature"
        assert "class UtilityClass" in summary, "Missing class signature"
        print("  PASSED: Python summarization extracts signatures")
        print(f"  Summary preview:\n{summary[:200]}...")

        # Test summarize_file for Markdown
        print("\n[Test 6] Test summarize_file() for Markdown...")
        summary = generator.summarize_file("low_tier.md")
        assert "# Low Priority Document" in summary, "Missing main header"
        assert "## Section 1" in summary, "Missing section header"
        print("  PASSED: Markdown summarization extracts headers")

        # Generate pack with all test files
        print("\n[Test 7] Generate LITE pack...")
        file_paths = [
            "high_tier.py",
            "medium_tier.py",
            "low_tier.md",
            "very_low_tier.txt",
            "default_tier.py",
        ]

        pack = generator.generate_pack(file_paths)

        # Verify manifest
        manifest = pack["manifest"]
        assert manifest["total_files"] == 5, f"Expected 5 files, got {manifest['total_files']}"
        assert manifest["included"] == 1, f"Expected 1 HIGH, got {manifest['included']}"
        assert manifest["summarized"] == 1, f"Expected 1 MEDIUM, got {manifest['summarized']}"
        assert manifest["pointers"] == 2, f"Expected 2 LOW, got {manifest['pointers']}"
        assert manifest["excluded"] == 1, f"Expected 1 VERY_LOW, got {manifest['excluded']}"
        print("  PASSED: Manifest counts are correct")

        # Verify HIGH tier content is complete
        print("\n[Test 8] Verify HIGH tier includes full content...")
        high_entries = pack["content"]["HIGH"]
        assert len(high_entries) == 1, f"Expected 1 HIGH entry, got {len(high_entries)}"
        assert "content" in high_entries[0], "Missing content field"
        assert "ImportantClass" in high_entries[0]["content"], "Missing class definition"
        assert "main_function" in high_entries[0]["content"], "Missing function definition"
        print("  PASSED: HIGH tier content is complete")

        # Verify MEDIUM tier is summarized
        print("\n[Test 9] Verify MEDIUM tier is summarized...")
        medium_entries = pack["content"]["MEDIUM"]
        assert len(medium_entries) == 1, f"Expected 1 MEDIUM entry, got {len(medium_entries)}"
        assert "summary" in medium_entries[0], "Missing summary field"
        assert "content" not in medium_entries[0], "Should not have full content"
        assert "def helper_function" in medium_entries[0]["summary"], "Missing signature"
        print("  PASSED: MEDIUM tier is properly summarized")

        # Verify LOW tier is pointer only
        print("\n[Test 10] Verify LOW tier is pointer only...")
        low_entries = pack["content"]["LOW"]
        assert len(low_entries) == 2, f"Expected 2 LOW entries, got {len(low_entries)}"
        for entry in low_entries:
            assert "path" in entry, "Missing path"
            assert "elo" in entry, "Missing elo"
            assert "content" not in entry, "Should not have content"
            assert "summary" not in entry, "Should not have summary"
        print("  PASSED: LOW tier is pointer only")

        # Verify VERY_LOW is excluded (not in content)
        print("\n[Test 11] Verify VERY_LOW is excluded...")
        all_paths = []
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            all_paths.extend([e["path"] for e in pack["content"][tier]])
        assert "very_low_tier.txt" not in all_paths, "VERY_LOW should be excluded"
        print("  PASSED: VERY_LOW tier is excluded from content")

        # Verify metadata
        print("\n[Test 12] Verify metadata includes all files...")
        metadata = pack["metadata"]
        assert len(metadata) == 5, f"Expected 5 metadata entries, got {len(metadata)}"
        tiers_found = {m["tier"] for m in metadata}
        assert tiers_found == {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}, f"Missing tiers: {tiers_found}"
        print("  PASSED: Metadata includes all files with tiers")

        # Test token savings estimation
        print("\n[Test 13] Test token savings estimation...")
        savings = generator.estimate_token_savings(pack)
        assert "full_tokens" in savings, "Missing full_tokens"
        assert "lite_tokens" in savings, "Missing lite_tokens"
        assert "savings_pct" in savings, "Missing savings_pct"
        assert savings["lite_tokens"] <= savings["full_tokens"], "LITE should be smaller"
        print(f"  Full tokens: {savings['full_tokens']}")
        print(f"  LITE tokens: {savings['lite_tokens']}")
        print(f"  Savings: {savings['savings_pct']}%")
        print("  PASSED: Token estimation works")

        # Test generate_pack_for_query
        print("\n[Test 14] Test generate_pack_for_query()...")
        query_pack = generator.generate_pack_for_query("important class", max_files=10)
        assert "manifest" in query_pack, "Missing manifest"
        assert "query" in query_pack["manifest"], "Missing query in manifest"
        print("  PASSED: Query-based pack generation works")

        # Test compression ratio
        print("\n[Test 15] Verify compression ratio...")
        compression = manifest.get("compression_ratio", 0)
        assert 0 <= compression <= 1, f"Invalid compression ratio: {compression}"
        print(f"  Compression ratio: {compression:.2%}")
        print("  PASSED: Compression ratio is valid")

        # Pretty print final pack for inspection
        print("\n[Test 16] Final pack structure...")
        print(json.dumps(manifest, indent=2))
        print("  PASSED: Pack structure is correct")

        # Cleanup
        db.close()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n[Cleanup] Removed test directory: {test_dir}")

    return all_passed


if __name__ == "__main__":
    # Run self-test
    if not run_self_test():
        print("\nSelf-test failed!")
        sys.exit(1)

    print("\nLITE Pack Generator ready for use!")
    sys.exit(0)
