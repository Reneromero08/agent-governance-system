#!/usr/bin/env python3
"""
ELO Database - SQLite-backed ELO scoring system for entity ranking.

Provides persistent storage and retrieval of ELO scores for:
- Vectors (content hashes)
- Files (file paths)
- Symbols (@SymbolName)
- ADRs (ADR-XXX)

Part of the agent-governance-system CORTEX infrastructure.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class EloDatabase:
    """SQLite-backed ELO score database for entity ranking."""

    VALID_TYPES = ["vector", "file", "symbol", "adr"]
    DEFAULT_ELO = 1000.0

    def __init__(self, db_path: str):
        """
        Initialize database connection, create tables if needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        # Ensure parent directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all required tables if they do not exist."""
        cursor = self.conn.cursor()

        # Vector ELO scores (for content hashes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_elo (
                entity_id TEXT PRIMARY KEY,
                elo_score REAL DEFAULT 1000,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT
            )
        """)

        # File ELO scores (for file paths)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_elo (
                entity_id TEXT PRIMARY KEY,
                elo_score REAL DEFAULT 1000,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT
            )
        """)

        # Symbol ELO scores (for @SymbolName)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_elo (
                entity_id TEXT PRIMARY KEY,
                elo_score REAL DEFAULT 1000,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT
            )
        """)

        # ADR ELO scores (for ADR-XXX)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adr_elo (
                entity_id TEXT PRIMARY KEY,
                elo_score REAL DEFAULT 1000,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT
            )
        """)

        self.conn.commit()

    def _get_table_name(self, entity_type: str) -> str:
        """
        Get the table name for a given entity type.

        Args:
            entity_type: One of VALID_TYPES.

        Returns:
            Table name string.

        Raises:
            ValueError: If entity_type is not valid.
        """
        if entity_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid entity_type: {entity_type}. "
                f"Must be one of {self.VALID_TYPES}"
            )
        return f"{entity_type}_elo"

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO8601 format."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    def get_elo(self, entity_type: str, entity_id: str) -> float:
        """
        Get ELO score for entity.

        Args:
            entity_type: One of VALID_TYPES.
            entity_id: Unique identifier for the entity.

        Returns:
            ELO score as float. Returns 1000.0 if not found.
        """
        table = self._get_table_name(entity_type)
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT elo_score FROM {table} WHERE entity_id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return self.DEFAULT_ELO
        return float(row["elo_score"])

    def set_elo(self, entity_type: str, entity_id: str, score: float) -> None:
        """
        Set ELO score for entity. Creates if not exists.

        Args:
            entity_type: One of VALID_TYPES.
            entity_id: Unique identifier for the entity.
            score: New ELO score value.
        """
        table = self._get_table_name(entity_type)
        timestamp = self._get_timestamp()
        cursor = self.conn.cursor()

        # Check if entity exists
        cursor.execute(
            f"SELECT entity_id FROM {table} WHERE entity_id = ?",
            (entity_id,)
        )
        exists = cursor.fetchone() is not None

        if exists:
            cursor.execute(
                f"UPDATE {table} SET elo_score = ? WHERE entity_id = ?",
                (score, entity_id)
            )
        else:
            cursor.execute(
                f"INSERT INTO {table} (entity_id, elo_score, access_count, last_accessed, created_at) "
                f"VALUES (?, ?, 0, ?, ?)",
                (entity_id, score, timestamp, timestamp)
            )

        self.conn.commit()

    def increment_access(self, entity_type: str, entity_id: str) -> None:
        """
        Increment access count and update last_accessed timestamp.

        Creates the entity with default ELO if it does not exist.

        Args:
            entity_type: One of VALID_TYPES.
            entity_id: Unique identifier for the entity.
        """
        table = self._get_table_name(entity_type)
        timestamp = self._get_timestamp()
        cursor = self.conn.cursor()

        # Check if entity exists
        cursor.execute(
            f"SELECT entity_id FROM {table} WHERE entity_id = ?",
            (entity_id,)
        )
        exists = cursor.fetchone() is not None

        if exists:
            cursor.execute(
                f"UPDATE {table} SET access_count = access_count + 1, last_accessed = ? "
                f"WHERE entity_id = ?",
                (timestamp, entity_id)
            )
        else:
            cursor.execute(
                f"INSERT INTO {table} (entity_id, elo_score, access_count, last_accessed, created_at) "
                f"VALUES (?, ?, 1, ?, ?)",
                (entity_id, self.DEFAULT_ELO, timestamp, timestamp)
            )

        self.conn.commit()

    def get_top_k(self, entity_type: str, k: int) -> List[Tuple[str, float]]:
        """
        Get top k entities by ELO score.

        Args:
            entity_type: One of VALID_TYPES.
            k: Number of top entities to return.

        Returns:
            List of (entity_id, elo_score) tuples, sorted by elo_score descending.
        """
        table = self._get_table_name(entity_type)
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT entity_id, elo_score FROM {table} ORDER BY elo_score DESC LIMIT ?",
            (k,)
        )
        rows = cursor.fetchall()
        return [(row["entity_id"], float(row["elo_score"])) for row in rows]

    def get_all_by_type(self, entity_type: str) -> List[Tuple[str, float, int, Optional[str]]]:
        """
        Get all entities of type.

        Args:
            entity_type: One of VALID_TYPES.

        Returns:
            List of (entity_id, elo_score, access_count, last_accessed) tuples.
        """
        table = self._get_table_name(entity_type)
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT entity_id, elo_score, access_count, last_accessed FROM {table}"
        )
        rows = cursor.fetchall()
        return [
            (row["entity_id"], float(row["elo_score"]), int(row["access_count"]), row["last_accessed"])
            for row in rows
        ]

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def run_self_test() -> bool:
    """
    Run comprehensive self-test of EloDatabase functionality.

    Returns:
        True if all tests pass, False otherwise.
    """
    import tempfile

    print("=" * 60)
    print("ELO Database Self-Test")
    print("=" * 60)

    # Use temporary database for testing
    test_db_path = os.path.join(tempfile.gettempdir(), "elo_test.db")

    # Clean up any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    all_passed = True

    try:
        # Test 1: Create database and tables
        print("\n[Test 1] Create database and tables...")
        db = EloDatabase(test_db_path)
        assert os.path.exists(test_db_path), "Database file not created"
        print("  PASSED: Database created successfully")

        # Verify all 4 tables exist
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {"vector_elo", "file_elo", "symbol_elo", "adr_elo"}
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"
        print("  PASSED: All 4 tables created")

        # Test 2: Insert test entities at different ELO levels
        print("\n[Test 2] Insert test entities at different ELO levels...")
        test_data = [
            ("vector", "hash_abc123", 1500.0),
            ("vector", "hash_def456", 1200.0),
            ("vector", "hash_ghi789", 800.0),
            ("file", "src/main.py", 1400.0),
            ("file", "tests/test_main.py", 1100.0),
            ("symbol", "@DatabaseManager", 1600.0),
            ("symbol", "@UserService", 1300.0),
            ("adr", "ADR-001", 1700.0),
            ("adr", "ADR-002", 1250.0),
        ]

        for entity_type, entity_id, score in test_data:
            db.set_elo(entity_type, entity_id, score)
        print(f"  PASSED: Inserted {len(test_data)} test entities")

        # Test 3: Query and verify get_elo works
        print("\n[Test 3] Query and verify get_elo works...")
        for entity_type, entity_id, expected_score in test_data:
            actual_score = db.get_elo(entity_type, entity_id)
            assert abs(actual_score - expected_score) < 0.001, \
                f"ELO mismatch for {entity_type}/{entity_id}: expected {expected_score}, got {actual_score}"
        print("  PASSED: All ELO scores retrieved correctly")

        # Test 4: Test default ELO for non-existent entity
        print("\n[Test 4] Test default ELO for non-existent entity...")
        default_score = db.get_elo("vector", "nonexistent_hash")
        assert default_score == 1000.0, f"Expected 1000.0, got {default_score}"
        print("  PASSED: Default ELO (1000.0) returned for non-existent entity")

        # Test 5: Update ELO and verify change
        print("\n[Test 5] Update ELO and verify change...")
        db.set_elo("vector", "hash_abc123", 1550.0)
        updated_score = db.get_elo("vector", "hash_abc123")
        assert abs(updated_score - 1550.0) < 0.001, f"Expected 1550.0, got {updated_score}"
        print("  PASSED: ELO update works correctly")

        # Test 6: Test get_top_k returns sorted results
        print("\n[Test 6] Test get_top_k returns sorted results...")
        top_vectors = db.get_top_k("vector", 3)
        assert len(top_vectors) == 3, f"Expected 3 results, got {len(top_vectors)}"
        assert top_vectors[0][0] == "hash_abc123", f"Expected hash_abc123 first, got {top_vectors[0][0]}"
        assert top_vectors[0][1] == 1550.0, f"Expected 1550.0, got {top_vectors[0][1]}"
        assert top_vectors[1][0] == "hash_def456", f"Expected hash_def456 second, got {top_vectors[1][0]}"
        assert top_vectors[2][0] == "hash_ghi789", f"Expected hash_ghi789 third, got {top_vectors[2][0]}"

        # Verify descending order
        for i in range(len(top_vectors) - 1):
            assert top_vectors[i][1] >= top_vectors[i + 1][1], "Results not in descending order"
        print("  PASSED: get_top_k returns sorted results (descending by ELO)")

        # Test 7: Test increment_access updates timestamp
        print("\n[Test 7] Test increment_access updates timestamp...")
        import time

        # Get initial state
        all_files = db.get_all_by_type("file")
        file_dict = {row[0]: row for row in all_files}
        initial_access = file_dict["src/main.py"][2]
        initial_timestamp = file_dict["src/main.py"][3]

        # Wait a moment and increment
        time.sleep(0.1)
        db.increment_access("file", "src/main.py")

        # Get updated state
        all_files = db.get_all_by_type("file")
        file_dict = {row[0]: row for row in all_files}
        new_access = file_dict["src/main.py"][2]
        new_timestamp = file_dict["src/main.py"][3]

        assert new_access == initial_access + 1, f"Access count not incremented: {initial_access} -> {new_access}"
        assert new_timestamp >= initial_timestamp, "Timestamp not updated"
        print(f"  PASSED: Access count incremented ({initial_access} -> {new_access})")
        print(f"  PASSED: Timestamp updated ({initial_timestamp} -> {new_timestamp})")

        # Test 8: Test increment_access creates new entity
        print("\n[Test 8] Test increment_access creates new entity...")
        db.increment_access("symbol", "@NewSymbol")
        new_symbol_elo = db.get_elo("symbol", "@NewSymbol")
        assert new_symbol_elo == 1000.0, f"Expected default ELO 1000.0, got {new_symbol_elo}"

        all_symbols = db.get_all_by_type("symbol")
        symbol_dict = {row[0]: row for row in all_symbols}
        assert "@NewSymbol" in symbol_dict, "New symbol not created"
        assert symbol_dict["@NewSymbol"][2] == 1, "Access count should be 1 for new entity"
        print("  PASSED: increment_access creates new entity with default ELO and access_count=1")

        # Test 9: Test invalid entity_type raises ValueError
        print("\n[Test 9] Test invalid entity_type raises ValueError...")
        try:
            db.get_elo("invalid_type", "test")
            print("  FAILED: Should have raised ValueError")
            all_passed = False
        except ValueError as e:
            assert "Invalid entity_type" in str(e)
            print("  PASSED: ValueError raised for invalid entity_type")

        # Test 10: Test get_all_by_type
        print("\n[Test 10] Test get_all_by_type...")
        all_adrs = db.get_all_by_type("adr")
        assert len(all_adrs) == 2, f"Expected 2 ADRs, got {len(all_adrs)}"
        adr_ids = {row[0] for row in all_adrs}
        assert adr_ids == {"ADR-001", "ADR-002"}, f"Unexpected ADR IDs: {adr_ids}"
        print("  PASSED: get_all_by_type returns all entities")

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
        all_passed = False
    finally:
        # Clean up test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

    return all_passed


if __name__ == "__main__":
    import sys

    # Run self-test first
    if not run_self_test():
        print("\nSelf-test failed!")
        sys.exit(1)

    # Now create the actual production database
    print("\n" + "=" * 60)
    print("Creating Production Database")
    print("=" * 60)

    # Get the script directory and navigate to the correct location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    db_path = os.path.join(repo_root, "NAVIGATION", "CORTEX", "_generated", "elo_scores.db")

    print(f"\nDatabase path: {db_path}")

    # Create production database
    db = EloDatabase(db_path)

    # Verify tables
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables created: {tables}")

    # Add some initial seed data for demonstration
    print("\nSeeding initial data...")

    # Seed some example entities
    seed_data = [
        ("adr", "ADR-001", 1200.0),  # Frequently referenced ADR
        ("adr", "ADR-002", 1100.0),  # Common ADR
        ("file", "CANON/CONSTITUTION.md", 1300.0),  # Core governance doc
        ("file", "NAVIGATION/CORTEX/README.md", 1150.0),  # Important readme
        ("symbol", "@EloDatabase", 1000.0),  # This class
    ]

    for entity_type, entity_id, score in seed_data:
        db.set_elo(entity_type, entity_id, score)
        print(f"  Added {entity_type}/{entity_id} with ELO={score}")

    db.close()

    print("\nProduction database ready!")
    print(f"Location: {db_path}")
    sys.exit(0)
