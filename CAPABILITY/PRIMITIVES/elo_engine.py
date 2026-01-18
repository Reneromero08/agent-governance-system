#!/usr/bin/env python3
"""
ELO Engine - Core ELO scoring algorithm for entity ranking.

Implements:
- Standard ELO update formula with adaptive K-factors
- Ebbinghaus forgetting curve decay
- Tier classification (HIGH, MEDIUM, LOW, VERY_LOW)
- Batch processing of search logs
- Append-only update logging

Part of the agent-governance-system CORTEX infrastructure.
"""

import json
import os
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Optional

# Import EloDatabase from sibling module
try:
    from .elo_db import EloDatabase
except ImportError:
    # Allow direct script execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from elo_db import EloDatabase


def _canonical_json_line(record: dict[str, Any]) -> bytes:
    """Convert a record to canonical JSON format with newline."""
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


class EloEngine:
    """
    Core ELO scoring engine for entity ranking.

    Processes search logs and updates ELO scores based on:
    - Selection/access patterns (ELO update formula)
    - Time decay (Ebbinghaus forgetting curve)
    """

    # Constants from specification
    HALF_LIFE = 30  # days for Ebbinghaus decay
    BASE_ELO = 800  # floor ELO score
    K_NEW = 16  # K-factor for new entities (access_count < 10)
    K_ESTABLISHED = 8  # K-factor for established entities
    ESTABLISHED_THRESHOLD = 10  # access count threshold

    def __init__(self, db: EloDatabase, updates_log_path: str):
        """
        Initialize with database and update log path.

        Args:
            db: EloDatabase instance for score storage
            updates_log_path: Path to elo_updates.jsonl log file
        """
        self.db = db
        self.updates_log_path = Path(updates_log_path)

        # Ensure parent directory exists
        self.updates_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def _log_update(
        self,
        entity_type: str,
        entity_id: str,
        elo_old: float,
        elo_new: float,
        reason: str
    ) -> None:
        """
        Append ELO update record to the updates log.

        Args:
            entity_type: Type of entity (vector, file, symbol, adr)
            entity_id: Unique identifier
            elo_old: Previous ELO score
            elo_new: New ELO score
            reason: Reason for update (e.g., semantic_search_rank_1, decay)
        """
        record = {
            "timestamp": self._get_timestamp(),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "elo_old": round(elo_old, 2),
            "elo_new": round(elo_new, 2),
            "delta": round(elo_new - elo_old, 2),
            "reason": reason
        }

        line = _canonical_json_line(record)

        # Use low-level file operations for append-only writes
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        fd = os.open(str(self.updates_log_path), flags, 0o644)
        try:
            written = 0
            while written < len(line):
                n = os.write(fd, line[written:])
                if n <= 0:
                    raise RuntimeError("failed to append ELO update entry")
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)

    def _get_k_factor(self, entity_type: str, entity_id: str) -> float:
        """
        Get the K-factor for an entity based on access count.

        New entities (access_count < 10) get K=16 for faster convergence.
        Established entities get K=8 for stability.

        Args:
            entity_type: Type of entity
            entity_id: Unique identifier

        Returns:
            K-factor (16 for new, 8 for established)
        """
        # Get all entities and find this one's access count
        all_entities = self.db.get_all_by_type(entity_type)
        for eid, _score, access_count, _last in all_entities:
            if eid == entity_id:
                if access_count < self.ESTABLISHED_THRESHOLD:
                    return self.K_NEW
                return self.K_ESTABLISHED
        # Entity doesn't exist yet, treat as new
        return self.K_NEW

    def update_elo(
        self,
        entity_type: str,
        entity_id: str,
        outcome: float,
        opponent_elo: float = 1000.0,
        reason: str = "access"
    ) -> float:
        """
        Update ELO score for entity based on match outcome.

        Uses the standard ELO formula:
        expected = 1 / (1 + 10^((opponent_elo - entity_elo) / 400))
        new_elo = old_elo + K * (outcome - expected)

        Args:
            entity_type: One of vector/file/symbol/adr
            entity_id: Unique identifier
            outcome: 1.0 for win (accessed), 0.0 for loss
            opponent_elo: ELO of comparison entity (default 1000)
            reason: Reason for update (for logging)

        Returns:
            New ELO score
        """
        # Get current ELO
        old_elo = self.db.get_elo(entity_type, entity_id)

        # Get K-factor based on entity's experience
        k = self._get_k_factor(entity_type, entity_id)

        # Calculate expected outcome
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - old_elo) / 400.0))

        # Calculate new ELO
        new_elo = old_elo + k * (outcome - expected)

        # Floor at BASE_ELO
        new_elo = max(new_elo, self.BASE_ELO)

        # Update database
        self.db.set_elo(entity_type, entity_id, new_elo)

        # Increment access count
        self.db.increment_access(entity_type, entity_id)

        # Log the update
        self._log_update(entity_type, entity_id, old_elo, new_elo, reason)

        return new_elo

    def decay_elo(
        self,
        entity_type: str,
        entity_id: str,
        days_since_access: float
    ) -> float:
        """
        Apply Ebbinghaus forgetting curve decay.

        Formula:
        retention = exp(-days_since_access / half_life)
        decayed_elo = base_elo + (current_elo - base_elo) * retention

        Args:
            entity_type: One of vector/file/symbol/adr
            entity_id: Unique identifier
            days_since_access: Days since last access

        Returns:
            Decayed ELO score (floors at BASE_ELO)
        """
        # Get current ELO
        current_elo = self.db.get_elo(entity_type, entity_id)

        # Calculate retention factor
        retention = exp(-days_since_access / self.HALF_LIFE)

        # Apply decay
        decayed_elo = self.BASE_ELO + (current_elo - self.BASE_ELO) * retention

        # Floor at BASE_ELO
        decayed_elo = max(decayed_elo, self.BASE_ELO)

        # Update database
        self.db.set_elo(entity_type, entity_id, decayed_elo)

        # Log the decay
        self._log_update(entity_type, entity_id, current_elo, decayed_elo, "decay")

        return decayed_elo

    def get_tier(self, elo: float) -> str:
        """
        Classify ELO into tier.

        Args:
            elo: ELO score

        Returns:
            Tier string: HIGH (>=1600), MEDIUM (>=1200), LOW (>=800), VERY_LOW (<800)
        """
        if elo >= 1600:
            return "HIGH"
        if elo >= 1200:
            return "MEDIUM"
        if elo >= 800:
            return "LOW"
        return "VERY_LOW"

    def process_search_log(
        self,
        log_path: str,
        processed_until: Optional[str] = None
    ) -> int:
        """
        Batch process search_log.jsonl entries.

        For each search result:
        - Rank 1 gets full boost (outcome=1.0)
        - Ranks 2-5 get partial boost (outcome=0.7)
        - Ranks 6+ get smaller boost (outcome=0.4)

        Args:
            log_path: Path to search_log.jsonl
            processed_until: ISO8601 timestamp to resume from (optional)

        Returns:
            Number of log entries processed
        """
        log_file = Path(log_path)
        if not log_file.exists():
            return 0

        processed_count = 0

        with open(log_file, "rb") as f:
            for line_bytes in f:
                if not line_bytes.strip():
                    continue

                try:
                    entry = json.loads(line_bytes.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                # Check if we should skip this entry based on timestamp
                entry_timestamp = entry.get("timestamp", "")
                if processed_until and entry_timestamp <= processed_until:
                    continue

                # Extract tool and results
                tool = entry.get("tool", "unknown")
                results = entry.get("results", [])

                # Process each result
                for result in results:
                    file_path = result.get("file_path")
                    rank = result.get("rank", 99)

                    if not file_path:
                        continue

                    # Determine outcome based on rank
                    if rank == 1:
                        outcome = 1.0
                    elif rank <= 5:
                        outcome = 0.7
                    else:
                        outcome = 0.4

                    # Build reason string
                    reason = f"{tool}_rank_{rank}"

                    # Update file ELO
                    self.update_elo("file", file_path, outcome, reason=reason)

                    # Also update vector ELO if hash is present
                    content_hash = result.get("hash")
                    if content_hash:
                        self.update_elo("vector", content_hash, outcome, reason=reason)

                processed_count += 1

        return processed_count

    def decay_all_stale(self, days_threshold: float = 7.0) -> int:
        """
        Apply decay to all entities not accessed in days_threshold days.

        Args:
            days_threshold: Minimum days since access to trigger decay

        Returns:
            Number of entities decayed
        """
        now = datetime.now(timezone.utc)
        decayed_count = 0

        for entity_type in self.db.VALID_TYPES:
            entities = self.db.get_all_by_type(entity_type)

            for entity_id, _score, _access_count, last_accessed in entities:
                if not last_accessed:
                    continue

                # Parse last_accessed timestamp
                try:
                    # Handle both formats: with and without timezone
                    if "+" in last_accessed or last_accessed.endswith("Z"):
                        last_dt = datetime.fromisoformat(
                            last_accessed.replace("Z", "+00:00")
                        )
                    else:
                        # Assume UTC if no timezone
                        last_dt = datetime.fromisoformat(last_accessed).replace(
                            tzinfo=timezone.utc
                        )
                except ValueError:
                    continue

                # Calculate days since access
                delta = now - last_dt
                days_since = delta.total_seconds() / (24 * 3600)

                if days_since >= days_threshold:
                    self.decay_elo(entity_type, entity_id, days_since)
                    decayed_count += 1

        return decayed_count


def run_self_test() -> bool:
    """
    Run comprehensive self-test of EloEngine functionality.

    Returns:
        True if all tests pass, False otherwise.
    """
    import tempfile
    import shutil

    print("=" * 60)
    print("ELO Engine Self-Test")
    print("=" * 60)

    # Create temporary directories for testing
    test_dir = tempfile.mkdtemp(prefix="elo_engine_test_")
    test_db_path = os.path.join(test_dir, "test_elo.db")
    test_log_path = os.path.join(test_dir, "elo_updates.jsonl")
    test_search_log_path = os.path.join(test_dir, "search_log.jsonl")

    all_passed = True

    try:
        # Test 1: Create engine
        print("\n[Test 1] Create EloEngine...")
        db = EloDatabase(test_db_path)
        engine = EloEngine(db, test_log_path)
        assert os.path.exists(test_db_path), "Database file not created"
        print("  PASSED: EloEngine created successfully")

        # Test 2: Update ELO with win
        print("\n[Test 2] Update ELO with win (outcome=1.0)...")
        initial_elo = db.get_elo("file", "test/file.py")
        assert initial_elo == 1000.0, f"Expected default 1000.0, got {initial_elo}"

        new_elo = engine.update_elo("file", "test/file.py", 1.0, reason="test_win")

        # For outcome=1.0 vs opponent=1000, expected=0.5, delta = K*(1-0.5) = K*0.5
        # With K=16 (new entity), delta should be 8
        expected_new = 1000.0 + 16 * 0.5  # 1008
        assert abs(new_elo - expected_new) < 0.1, f"Expected ~{expected_new}, got {new_elo}"
        print(f"  PASSED: ELO updated from 1000.0 to {new_elo:.2f}")

        # Test 3: Update ELO with loss
        print("\n[Test 3] Update ELO with loss (outcome=0.0)...")
        # Set a known starting ELO
        db.set_elo("file", "test/loser.py", 1200.0)
        new_elo = engine.update_elo("file", "test/loser.py", 0.0, opponent_elo=1000.0, reason="test_loss")

        # For outcome=0.0 vs opponent=1000, expected = 1/(1+10^((1000-1200)/400)) = 0.76
        # delta = K*(0-0.76) = -12.16 with K=16
        assert new_elo < 1200.0, f"ELO should decrease on loss, got {new_elo}"
        print(f"  PASSED: ELO decreased from 1200.0 to {new_elo:.2f}")

        # Test 4: Test decay function
        print("\n[Test 4] Test Ebbinghaus decay...")
        db.set_elo("file", "test/decay.py", 1600.0)

        # Decay with 30 days (half-life)
        decayed = engine.decay_elo("file", "test/decay.py", 30.0)

        # At half-life, retention = exp(-1) ~= 0.368
        # decayed = 800 + (1600 - 800) * 0.368 = 800 + 294.4 = 1094.4
        expected_decay = 800 + (1600 - 800) * exp(-1)
        assert abs(decayed - expected_decay) < 0.1, f"Expected ~{expected_decay:.2f}, got {decayed:.2f}"
        print(f"  PASSED: ELO decayed from 1600.0 to {decayed:.2f} (expected ~{expected_decay:.2f})")

        # Test 5: Test decay floors at BASE_ELO
        print("\n[Test 5] Test decay floors at BASE_ELO...")
        db.set_elo("file", "test/floor.py", 850.0)

        # Decay with 365 days (very long)
        decayed = engine.decay_elo("file", "test/floor.py", 365.0)

        assert decayed >= engine.BASE_ELO, f"Decay should floor at {engine.BASE_ELO}, got {decayed}"
        print(f"  PASSED: ELO floored at {decayed:.2f} (BASE_ELO={engine.BASE_ELO})")

        # Test 6: Tier classification
        print("\n[Test 6] Test tier classification...")
        test_cases = [
            (1700.0, "HIGH"),
            (1600.0, "HIGH"),
            (1400.0, "MEDIUM"),
            (1200.0, "MEDIUM"),
            (1000.0, "LOW"),
            (800.0, "LOW"),
            (700.0, "VERY_LOW"),
        ]
        for elo, expected_tier in test_cases:
            tier = engine.get_tier(elo)
            assert tier == expected_tier, f"ELO {elo}: expected {expected_tier}, got {tier}"
        print("  PASSED: All tier boundaries correct")

        # Test 7: Process search log
        print("\n[Test 7] Process search log...")

        # Create sample search log entries
        search_entries = []
        for i in range(10):
            entry = {
                "session_id": f"session_{i}",
                "timestamp": f"2025-01-{10+i:02d}T12:00:00+00:00",
                "tool": "semantic_search" if i % 2 == 0 else "grep_search",
                "query": f"test query {i}",
                "results": [
                    {"hash": f"hash_{i}_1", "file_path": f"src/module_{i}.py", "rank": 1, "similarity": 0.95},
                    {"hash": f"hash_{i}_2", "file_path": f"src/util_{i}.py", "rank": 2, "similarity": 0.85},
                    {"hash": f"hash_{i}_3", "file_path": f"docs/doc_{i}.md", "rank": 6, "similarity": 0.65},
                ]
            }
            search_entries.append(entry)

        # Write search log
        with open(test_search_log_path, "w", encoding="utf-8") as f:
            for entry in search_entries:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

        # Process the log
        processed = engine.process_search_log(test_search_log_path)
        assert processed == 10, f"Expected 10 entries processed, got {processed}"
        print(f"  PASSED: Processed {processed} log entries")

        # Test 8: Verify ELO changes from search log processing
        print("\n[Test 8] Verify ELO changes from search log...")

        # Check that rank 1 files got the biggest boost
        file_0_elo = db.get_elo("file", "src/module_0.py")
        file_0_util_elo = db.get_elo("file", "src/util_0.py")
        file_0_doc_elo = db.get_elo("file", "docs/doc_0.md")

        # Rank 1 should have highest ELO boost
        assert file_0_elo > file_0_util_elo, "Rank 1 should have higher ELO than rank 2"
        assert file_0_util_elo > file_0_doc_elo, "Rank 2 should have higher ELO than rank 6"
        print(f"  PASSED: Rank ordering preserved (rank1={file_0_elo:.2f}, rank2={file_0_util_elo:.2f}, rank6={file_0_doc_elo:.2f})")

        # Test 9: Verify elo_updates.jsonl logging
        print("\n[Test 9] Verify elo_updates.jsonl logging...")

        assert os.path.exists(test_log_path), "elo_updates.jsonl not created"

        with open(test_log_path, "r", encoding="utf-8") as f:
            log_lines = f.readlines()

        assert len(log_lines) > 0, "elo_updates.jsonl is empty"

        # Validate log entry structure
        required_fields = {"timestamp", "entity_type", "entity_id", "elo_old", "elo_new", "delta", "reason"}
        first_entry = json.loads(log_lines[0])
        missing = required_fields - set(first_entry.keys())
        assert not missing, f"Log entry missing fields: {missing}"
        print(f"  PASSED: {len(log_lines)} updates logged with correct schema")

        # Test 10: Test process_search_log with resume timestamp
        print("\n[Test 10] Test process_search_log with resume timestamp...")

        # Process only entries after a certain timestamp
        processed = engine.process_search_log(
            test_search_log_path,
            processed_until="2025-01-15T12:00:00+00:00"
        )

        # Should skip entries 0-5 (timestamps 2025-01-10 through 2025-01-15)
        assert processed == 4, f"Expected 4 entries after resume, got {processed}"
        print(f"  PASSED: Resumed processing correctly (skipped older entries)")

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
    print("Demo: EloEngine with Production Database")
    print("=" * 60)

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    db_path = os.path.join(repo_root, "NAVIGATION", "CORTEX", "_generated", "elo_scores.db")
    updates_log_path = os.path.join(repo_root, "NAVIGATION", "CORTEX", "_generated", "elo_updates.jsonl")
    search_log_path = os.path.join(repo_root, "NAVIGATION", "CORTEX", "_generated", "search_log.jsonl")

    print(f"\nDatabase path: {db_path}")
    print(f"Updates log path: {updates_log_path}")
    print(f"Search log path: {search_log_path}")

    # Create engine
    db = EloDatabase(db_path)
    engine = EloEngine(db, updates_log_path)

    # Show current top files
    print("\nTop 5 files by ELO:")
    for file_path, score in db.get_top_k("file", 5):
        tier = engine.get_tier(score)
        print(f"  {tier:8s} {score:7.2f}  {file_path}")

    # Process any pending search logs
    if os.path.exists(search_log_path):
        processed = engine.process_search_log(search_log_path)
        print(f"\nProcessed {processed} search log entries")
    else:
        print(f"\nNo search log found at {search_log_path}")

    db.close()
    print("\nDemo complete!")
    sys.exit(0)
