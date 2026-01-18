#!/usr/bin/env python3
"""
Memory Pruning - ELO-based memory pruning for bounded short-term memory growth.

Implements pruning policy based on ELO tier and staleness:
- VERY_LOW (<800) + 30 days stale: Archive to MEMORY/ARCHIVE/pruned/
- LOW (800-1199) + 14 days stale: Flag for compression (summary)
- MEDIUM (1200-1599): Retain
- HIGH (1600+): Never prune (protected)

Part of the agent-governance-system CORTEX infrastructure.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import EloDatabase and EloEngine from sibling module
try:
    from ..PRIMITIVES.elo_db import EloDatabase
    from ..PRIMITIVES.elo_engine import EloEngine
except ImportError:
    # Allow direct script execution
    import sys
    primitives_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "PRIMITIVES"
    )
    sys.path.insert(0, primitives_path)
    from elo_db import EloDatabase
    from elo_engine import EloEngine


def _canonical_json_line(record: Dict[str, Any]) -> bytes:
    """Convert a record to canonical JSON format with newline."""
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


class MemoryPruner:
    """
    ELO-based memory pruning for bounded short-term memory growth.

    Scans short-term memory directories and applies pruning policy:
    - VERY_LOW ELO + 30 days stale -> Archive
    - LOW ELO + 14 days stale -> Flag for compression
    - MEDIUM/HIGH ELO -> Retain
    """

    ARCHIVE_DIR = "MEMORY/ARCHIVE/pruned"

    # Short-term memory directories to scan (relative to repo root)
    PRUNABLE_DIRS = [
        "INBOX/reports",
        "THOUGHT/LAB/*/scratch",
        "NAVIGATION/CORTEX/_generated",
    ]

    # Files to never prune (regardless of ELO)
    PROTECTED_FILES = [
        "elo_scores.db",
        "manifest.jsonl",
    ]

    # Policy thresholds
    VERY_LOW_STALE_DAYS = 30
    LOW_STALE_DAYS = 14

    def __init__(self, repo_root: str, db: EloDatabase, engine: EloEngine):
        """
        Initialize pruner with repo root, database, and engine.

        Args:
            repo_root: Absolute path to the repository root.
            db: EloDatabase instance for ELO score lookups.
            engine: EloEngine instance for tier classification.
        """
        self.repo_root = Path(repo_root).resolve()
        self.db = db
        self.engine = engine
        self.archive_dir = self.repo_root / self.ARCHIVE_DIR

    def _get_file_elo(self, file_path: Path) -> float:
        """
        Get ELO score for a file.

        Args:
            file_path: Path to the file (relative or absolute).

        Returns:
            ELO score (defaults to 1000.0 for unknown files).
        """
        # Normalize to relative path from repo root
        try:
            rel_path = file_path.relative_to(self.repo_root)
        except ValueError:
            rel_path = file_path

        # Convert to forward-slash path for consistency
        rel_path_str = str(rel_path).replace("\\", "/")

        return self.db.get_elo("file", rel_path_str)

    def _get_days_stale(self, file_path: Path) -> float:
        """
        Calculate days since file was last accessed.

        Uses file modification time as the access indicator.

        Args:
            file_path: Path to the file.

        Returns:
            Days since last access/modification.
        """
        if not file_path.exists():
            return 0.0

        # Get file modification time
        mtime = file_path.stat().st_mtime
        mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)

        # Calculate days since modification
        now = datetime.now(timezone.utc)
        delta = now - mtime_dt
        return delta.total_seconds() / (24 * 3600)

    def _is_protected(self, file_path: Path) -> bool:
        """
        Check if a file is protected from pruning.

        Args:
            file_path: Path to the file.

        Returns:
            True if file should never be pruned.
        """
        return file_path.name in self.PROTECTED_FILES

    def _scan_prunable_files(self) -> List[Path]:
        """
        Scan all prunable directories and return list of files.

        Returns:
            List of Path objects for all files in prunable directories.
        """
        files = []

        for dir_pattern in self.PRUNABLE_DIRS:
            # Handle glob patterns
            if "*" in dir_pattern:
                # Use glob to find matching directories
                pattern_path = self.repo_root / dir_pattern
                parent = pattern_path.parent
                if parent.exists():
                    # Find all matching directories
                    for match_dir in parent.glob(pattern_path.name):
                        if match_dir.is_dir():
                            for f in match_dir.iterdir():
                                if f.is_file() and not self._is_protected(f):
                                    files.append(f)
            else:
                # Direct directory
                dir_path = self.repo_root / dir_pattern
                if dir_path.exists() and dir_path.is_dir():
                    for f in dir_path.iterdir():
                        if f.is_file() and not self._is_protected(f):
                            files.append(f)

        return files

    def list_prune_candidates(self) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Scan short-term memory directories and return candidates.

        Returns:
            {
                "archive": [(path, elo, days_stale), ...],  # VERY_LOW + 30 days
                "compress": [(path, elo, days_stale), ...], # LOW + 14 days
                "retain": [(path, elo, days_stale), ...]    # MEDIUM+
            }
        """
        candidates = {
            "archive": [],
            "compress": [],
            "retain": [],
        }

        files = self._scan_prunable_files()

        for file_path in files:
            elo = self._get_file_elo(file_path)
            days_stale = self._get_days_stale(file_path)
            tier = self.engine.get_tier(elo)

            # Get relative path for display
            try:
                rel_path = str(file_path.relative_to(self.repo_root))
            except ValueError:
                rel_path = str(file_path)

            # Convert to forward-slash path
            rel_path = rel_path.replace("\\", "/")

            entry = (rel_path, elo, days_stale)

            # Apply policy
            if tier == "HIGH":
                # Never prune HIGH ELO content
                candidates["retain"].append(entry)
            elif tier == "VERY_LOW" and days_stale >= self.VERY_LOW_STALE_DAYS:
                # Archive very low ELO + 30 days stale
                candidates["archive"].append(entry)
            elif tier == "LOW" and days_stale >= self.LOW_STALE_DAYS:
                # Flag for compression: LOW ELO + 14 days stale
                candidates["compress"].append(entry)
            else:
                # Retain everything else
                candidates["retain"].append(entry)

        return candidates

    def archive_file(self, file_path: str) -> str:
        """
        Move file to MEMORY/ARCHIVE/pruned/ with manifest.

        - Creates ARCHIVE_DIR if needed
        - Moves file preserving relative path structure
        - Updates manifest.jsonl with archive record

        Args:
            file_path: Relative path from repo root.

        Returns:
            Archive path (relative to repo root).
        """
        # Resolve full paths
        source_path = self.repo_root / file_path
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Create archive structure mirroring original path
        archive_path = self.archive_dir / file_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # Get ELO and staleness before moving
        elo = self._get_file_elo(source_path)
        days_stale = self._get_days_stale(source_path)

        # Move the file
        shutil.move(str(source_path), str(archive_path))

        # Update manifest
        manifest_path = self.archive_dir / "manifest.jsonl"
        manifest_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_path": file_path.replace("\\", "/"),
            "archive_path": str(archive_path.relative_to(self.repo_root)).replace("\\", "/"),
            "elo_score": round(elo, 2),
            "days_stale": round(days_stale, 2),
            "reason": "VERY_LOW_ELO_STALE",
        }

        line = _canonical_json_line(manifest_record)

        # Append to manifest using low-level operations for safety
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        fd = os.open(str(manifest_path), flags, 0o644)
        try:
            written = 0
            while written < len(line):
                n = os.write(fd, line[written:])
                if n <= 0:
                    raise RuntimeError("Failed to append manifest entry")
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)

        # Return relative archive path
        return str(archive_path.relative_to(self.repo_root)).replace("\\", "/")

    def execute_pruning(self, dry_run: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute pruning based on policy.

        Args:
            dry_run: If True, only report what would be pruned.

        Returns:
            {
                "archived": [{"path": ..., "archive_path": ..., "elo": ...}, ...],
                "flagged_compress": [...],
                "retained": [...],
                "errors": [...]
            }
        """
        result = {
            "archived": [],
            "flagged_compress": [],
            "retained": [],
            "errors": [],
        }

        candidates = self.list_prune_candidates()

        # Process archive candidates
        for path, elo, days_stale in candidates["archive"]:
            entry = {
                "path": path,
                "elo": round(elo, 2),
                "days_stale": round(days_stale, 2),
                "tier": self.engine.get_tier(elo),
            }

            if dry_run:
                entry["action"] = "would_archive"
                result["archived"].append(entry)
            else:
                try:
                    archive_path = self.archive_file(path)
                    entry["archive_path"] = archive_path
                    entry["action"] = "archived"
                    result["archived"].append(entry)
                except Exception as e:
                    result["errors"].append({
                        "path": path,
                        "error": str(e),
                        "action": "archive_failed",
                    })

        # Process compress candidates (flag only, actual compression is separate)
        for path, elo, days_stale in candidates["compress"]:
            entry = {
                "path": path,
                "elo": round(elo, 2),
                "days_stale": round(days_stale, 2),
                "tier": self.engine.get_tier(elo),
                "action": "flagged_for_compression" if not dry_run else "would_flag_compress",
            }
            result["flagged_compress"].append(entry)

        # Process retain candidates
        for path, elo, days_stale in candidates["retain"]:
            entry = {
                "path": path,
                "elo": round(elo, 2),
                "days_stale": round(days_stale, 2),
                "tier": self.engine.get_tier(elo),
                "action": "retained",
            }
            result["retained"].append(entry)

        return result

    def generate_report(self) -> str:
        """
        Generate human-readable pruning report.

        Returns:
            Formatted report string.
        """
        candidates = self.list_prune_candidates()

        lines = [
            "=" * 60,
            "MEMORY PRUNING REPORT",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Repository: {self.repo_root}",
            "",
        ]

        # Summary
        total_archive = len(candidates["archive"])
        total_compress = len(candidates["compress"])
        total_retain = len(candidates["retain"])
        total_files = total_archive + total_compress + total_retain

        lines.extend([
            "SUMMARY",
            "-" * 40,
            f"Total files scanned: {total_files}",
            f"  - To archive (VERY_LOW + 30d): {total_archive}",
            f"  - To compress (LOW + 14d): {total_compress}",
            f"  - To retain (MEDIUM/HIGH): {total_retain}",
            "",
        ])

        # Archive candidates
        if candidates["archive"]:
            lines.extend([
                "ARCHIVE CANDIDATES (VERY_LOW ELO + 30+ days stale)",
                "-" * 40,
            ])
            for path, elo, days in candidates["archive"]:
                tier = self.engine.get_tier(elo)
                lines.append(f"  [{tier:8s}] ELO={elo:7.2f} | {days:5.1f}d stale | {path}")
            lines.append("")

        # Compress candidates
        if candidates["compress"]:
            lines.extend([
                "COMPRESSION CANDIDATES (LOW ELO + 14+ days stale)",
                "-" * 40,
            ])
            for path, elo, days in candidates["compress"]:
                tier = self.engine.get_tier(elo)
                lines.append(f"  [{tier:8s}] ELO={elo:7.2f} | {days:5.1f}d stale | {path}")
            lines.append("")

        # Retained files (high ELO or recent)
        if candidates["retain"]:
            lines.extend([
                "RETAINED FILES (MEDIUM+ ELO or recently accessed)",
                "-" * 40,
            ])
            for path, elo, days in candidates["retain"]:
                tier = self.engine.get_tier(elo)
                lines.append(f"  [{tier:8s}] ELO={elo:7.2f} | {days:5.1f}d stale | {path}")
            lines.append("")

        lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        return "\n".join(lines)


def run_self_test() -> bool:
    """
    Run comprehensive self-test of MemoryPruner functionality.

    Returns:
        True if all tests pass, False otherwise.
    """
    import tempfile

    print("=" * 60)
    print("Memory Pruner Self-Test")
    print("=" * 60)

    # Create temporary test directory structure
    test_dir = tempfile.mkdtemp(prefix="memory_pruner_test_")
    test_repo = Path(test_dir)

    all_passed = True

    try:
        # Create test directory structure
        print("\n[Setup] Creating test directory structure...")

        # Create prunable directories
        (test_repo / "INBOX" / "reports").mkdir(parents=True)
        (test_repo / "THOUGHT" / "LAB" / "FORMULA" / "scratch").mkdir(parents=True)
        (test_repo / "THOUGHT" / "LAB" / "EXPERIMENT" / "scratch").mkdir(parents=True)
        (test_repo / "NAVIGATION" / "CORTEX" / "_generated").mkdir(parents=True)
        (test_repo / "MEMORY" / "ARCHIVE" / "pruned").mkdir(parents=True)

        # Create test files with different ages
        import time

        test_files = [
            # Very stale files (simulate 45 days old)
            "INBOX/reports/old_session_001.md",
            "INBOX/reports/old_session_002.md",
            # Moderately stale files (simulate 20 days old)
            "INBOX/reports/moderate_session_003.md",
            "THOUGHT/LAB/FORMULA/scratch/stale_scratch.txt",
            # Recent files
            "INBOX/reports/recent_session_004.md",
            "NAVIGATION/CORTEX/_generated/session_audit.jsonl",
            # Protected file (should never be pruned)
            "NAVIGATION/CORTEX/_generated/elo_scores.db",
        ]

        for file_path in test_files:
            full_path = test_repo / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"Test content for {file_path}")

        print(f"  Created {len(test_files)} test files")

        # Test 1: Create database and engine
        print("\n[Test 1] Create EloDatabase and EloEngine...")
        db_path = str(test_repo / "test_elo.db")
        updates_log_path = str(test_repo / "elo_updates.jsonl")

        db = EloDatabase(db_path)
        engine = EloEngine(db, updates_log_path)
        print("  PASSED: Database and engine created")

        # Test 2: Set up ELO scores for test files
        print("\n[Test 2] Set up ELO scores...")
        # Very low ELO files (should be archived if stale)
        db.set_elo("file", "INBOX/reports/old_session_001.md", 750.0)
        db.set_elo("file", "INBOX/reports/old_session_002.md", 700.0)
        # Low ELO files (should be flagged for compression if stale)
        db.set_elo("file", "INBOX/reports/moderate_session_003.md", 900.0)
        db.set_elo("file", "THOUGHT/LAB/FORMULA/scratch/stale_scratch.txt", 850.0)
        # Medium ELO (should be retained)
        db.set_elo("file", "INBOX/reports/recent_session_004.md", 1300.0)
        # High ELO (should never be pruned)
        db.set_elo("file", "NAVIGATION/CORTEX/_generated/session_audit.jsonl", 1700.0)
        print("  PASSED: ELO scores set")

        # Test 3: Create pruner
        print("\n[Test 3] Create MemoryPruner...")
        pruner = MemoryPruner(str(test_repo), db, engine)
        print("  PASSED: MemoryPruner created")

        # Test 4: Verify tier classification
        print("\n[Test 4] Verify tier classification...")
        assert engine.get_tier(750.0) == "VERY_LOW", "750 should be VERY_LOW"
        assert engine.get_tier(900.0) == "LOW", "900 should be LOW"
        assert engine.get_tier(1300.0) == "MEDIUM", "1300 should be MEDIUM"
        assert engine.get_tier(1700.0) == "HIGH", "1700 should be HIGH"
        print("  PASSED: Tier classification correct")

        # Test 5: Modify file times to simulate staleness
        print("\n[Test 5] Simulate file staleness...")
        import os as os_module

        # Set old files to 45 days ago
        very_old_time = time.time() - (45 * 24 * 3600)
        for old_file in ["INBOX/reports/old_session_001.md", "INBOX/reports/old_session_002.md"]:
            full_path = test_repo / old_file
            os_module.utime(str(full_path), (very_old_time, very_old_time))

        # Set moderate files to 20 days ago
        moderate_time = time.time() - (20 * 24 * 3600)
        for mod_file in ["INBOX/reports/moderate_session_003.md", "THOUGHT/LAB/FORMULA/scratch/stale_scratch.txt"]:
            full_path = test_repo / mod_file
            os_module.utime(str(full_path), (moderate_time, moderate_time))

        print("  PASSED: File times modified")

        # Test 6: List prune candidates
        print("\n[Test 6] List prune candidates...")
        candidates = pruner.list_prune_candidates()

        print(f"  Archive candidates: {len(candidates['archive'])}")
        print(f"  Compress candidates: {len(candidates['compress'])}")
        print(f"  Retain candidates: {len(candidates['retain'])}")

        # Verify archive candidates (VERY_LOW + 30+ days)
        archive_paths = [c[0] for c in candidates["archive"]]
        assert "INBOX/reports/old_session_001.md" in archive_paths, "old_session_001 should be archived"
        assert "INBOX/reports/old_session_002.md" in archive_paths, "old_session_002 should be archived"
        print("  PASSED: Archive candidates identified correctly")

        # Verify compress candidates (LOW + 14+ days)
        compress_paths = [c[0] for c in candidates["compress"]]
        assert "INBOX/reports/moderate_session_003.md" in compress_paths, "moderate_session_003 should be compressed"
        print("  PASSED: Compress candidates identified correctly")

        # Verify retained files
        retain_paths = [c[0] for c in candidates["retain"]]
        assert "INBOX/reports/recent_session_004.md" in retain_paths, "recent_session_004 should be retained"
        assert "NAVIGATION/CORTEX/_generated/session_audit.jsonl" in retain_paths, "HIGH ELO file should be retained"
        print("  PASSED: Retain candidates identified correctly")

        # Verify protected files are not in any list
        all_candidates = archive_paths + compress_paths + retain_paths
        assert "NAVIGATION/CORTEX/_generated/elo_scores.db" not in all_candidates, "Protected file should not be scanned"
        print("  PASSED: Protected files excluded")

        # Test 7: Execute pruning (dry run)
        print("\n[Test 7] Execute pruning (dry_run=True)...")
        dry_result = pruner.execute_pruning(dry_run=True)

        assert len(dry_result["archived"]) == 2, f"Expected 2 archive candidates, got {len(dry_result['archived'])}"
        assert all(e["action"] == "would_archive" for e in dry_result["archived"]), "Dry run should report would_archive"
        assert len(dry_result["errors"]) == 0, f"Unexpected errors: {dry_result['errors']}"

        # Verify files still exist (dry run shouldn't move them)
        assert (test_repo / "INBOX/reports/old_session_001.md").exists(), "File should still exist after dry run"
        print("  PASSED: Dry run completed without moving files")

        # Test 8: Execute pruning (actual)
        print("\n[Test 8] Execute pruning (dry_run=False)...")
        actual_result = pruner.execute_pruning(dry_run=False)

        assert len(actual_result["archived"]) == 2, f"Expected 2 archived files, got {len(actual_result['archived'])}"
        assert all(e["action"] == "archived" for e in actual_result["archived"]), "Should report archived"
        assert len(actual_result["errors"]) == 0, f"Unexpected errors: {actual_result['errors']}"
        print("  PASSED: Actual pruning completed")

        # Test 9: Verify files were archived
        print("\n[Test 9] Verify files were archived...")

        # Original files should be gone
        assert not (test_repo / "INBOX/reports/old_session_001.md").exists(), "Original file should be moved"
        assert not (test_repo / "INBOX/reports/old_session_002.md").exists(), "Original file should be moved"

        # Archive files should exist
        archive_1 = test_repo / "MEMORY/ARCHIVE/pruned/INBOX/reports/old_session_001.md"
        archive_2 = test_repo / "MEMORY/ARCHIVE/pruned/INBOX/reports/old_session_002.md"
        assert archive_1.exists(), f"Archive file should exist: {archive_1}"
        assert archive_2.exists(), f"Archive file should exist: {archive_2}"
        print("  PASSED: Files moved to archive correctly")

        # Test 10: Verify manifest was updated
        print("\n[Test 10] Verify manifest was updated...")
        manifest_path = test_repo / "MEMORY/ARCHIVE/pruned/manifest.jsonl"
        assert manifest_path.exists(), "Manifest file should exist"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_lines = f.readlines()

        assert len(manifest_lines) == 2, f"Expected 2 manifest entries, got {len(manifest_lines)}"

        # Validate manifest entry structure
        required_fields = {"timestamp", "original_path", "archive_path", "elo_score", "days_stale", "reason"}
        entry = json.loads(manifest_lines[0])
        missing = required_fields - set(entry.keys())
        assert not missing, f"Manifest entry missing fields: {missing}"
        print("  PASSED: Manifest updated with correct schema")

        # Test 11: Generate report
        print("\n[Test 11] Generate pruning report...")
        report = pruner.generate_report()

        assert "MEMORY PRUNING REPORT" in report, "Report should have header"
        assert "Total files scanned" in report, "Report should have summary"
        print("  PASSED: Report generated successfully")
        print("\n--- Report Preview ---")
        # Print first 20 lines of report
        for line in report.split("\n")[:20]:
            print(f"  {line}")
        print("  ...")

        # Test 12: Verify HIGH ELO files never pruned
        print("\n[Test 12] Verify HIGH ELO files never pruned...")
        # Make the high ELO file very old
        high_elo_file = test_repo / "NAVIGATION/CORTEX/_generated/session_audit.jsonl"
        very_old_time = time.time() - (100 * 24 * 3600)  # 100 days old
        os_module.utime(str(high_elo_file), (very_old_time, very_old_time))

        # Re-check candidates
        candidates = pruner.list_prune_candidates()
        archive_paths = [c[0] for c in candidates["archive"]]
        assert "NAVIGATION/CORTEX/_generated/session_audit.jsonl" not in archive_paths, "HIGH ELO should never be archived"

        retain_paths = [c[0] for c in candidates["retain"]]
        assert "NAVIGATION/CORTEX/_generated/session_audit.jsonl" in retain_paths, "HIGH ELO should always be retained"
        print("  PASSED: HIGH ELO content is protected")

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
    import sys

    # Run self-test
    if not run_self_test():
        print("\nSelf-test failed!")
        sys.exit(1)

    # Demo with production database
    print("\n" + "=" * 60)
    print("Demo: MemoryPruner with Production Database")
    print("=" * 60)

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    db_path = os.path.join(repo_root, "NAVIGATION", "CORTEX", "_generated", "elo_scores.db")
    updates_log_path = os.path.join(repo_root, "NAVIGATION", "CORTEX", "_generated", "elo_updates.jsonl")

    print(f"\nRepository root: {repo_root}")
    print(f"Database path: {db_path}")

    # Create pruner
    db = EloDatabase(db_path)
    engine = EloEngine(db, updates_log_path)
    pruner = MemoryPruner(repo_root, db, engine)

    # Generate and print report
    print("\n" + pruner.generate_report())

    # Show what would be pruned (dry run)
    print("\n" + "=" * 60)
    print("DRY RUN RESULTS")
    print("=" * 60)

    result = pruner.execute_pruning(dry_run=True)

    print(f"\nWould archive: {len(result['archived'])} files")
    for entry in result["archived"][:5]:  # Show first 5
        print(f"  - {entry['path']} (ELO={entry['elo']}, {entry['days_stale']:.1f}d stale)")

    print(f"\nWould flag for compression: {len(result['flagged_compress'])} files")
    for entry in result["flagged_compress"][:5]:  # Show first 5
        print(f"  - {entry['path']} (ELO={entry['elo']}, {entry['days_stale']:.1f}d stale)")

    print(f"\nWould retain: {len(result['retained'])} files")

    db.close()
    print("\nDemo complete!")
    sys.exit(0)
