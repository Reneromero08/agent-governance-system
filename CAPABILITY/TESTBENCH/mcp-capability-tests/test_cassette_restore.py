#!/usr/bin/env python3
"""
Phase 6 Tests: Cassette Restore Guarantee

Tests for export/import cartridge, corrupt-and-restore, and Merkle verification.
"""

import os
import sys
import json
import shutil
import uuid
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "network"))
sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "PRIMITIVES"))

import pytest

# Use a test directory within the project to avoid GuardedWriter path issues
TEST_BASE_DIR = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "cassettes" / "_test_phase6"


def get_test_dir():
    """Create a unique test directory within the project."""
    test_id = str(uuid.uuid4())[:8]
    temp_path = TEST_BASE_DIR / test_id
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


class TestCartridgeExportImport:
    """Test cartridge export and import functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests within project."""
        temp_path = get_test_dir()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def memory_cassette(self, temp_dir):
        """Create a test memory cassette."""
        from memory_cassette import MemoryCassette

        db_path = temp_dir / "test_cassette.db"
        cassette = MemoryCassette(db_path=db_path, agent_id="test_agent")
        return cassette

    def test_export_empty_cartridge(self, memory_cassette, temp_dir):
        """Test exporting an empty cassette."""
        export_dir = temp_dir / "export"

        manifest = memory_cassette.export_cartridge(export_dir)

        assert manifest["record_count"] == 0
        assert manifest["receipt_count"] == 0
        assert (export_dir / "manifest.json").exists()
        assert (export_dir / "records.jsonl").exists()
        assert (export_dir / "receipts.jsonl").exists()

    def test_export_with_memories(self, memory_cassette, temp_dir):
        """Test exporting cassette with memories."""
        # Save some memories
        hash1, receipt1 = memory_cassette.memory_save("First memory content")
        hash2, receipt2 = memory_cassette.memory_save("Second memory content")
        hash3, receipt3 = memory_cassette.memory_save("Third memory content")

        export_dir = temp_dir / "export"
        manifest = memory_cassette.export_cartridge(export_dir)

        assert manifest["record_count"] == 3
        assert manifest["content_merkle_root"] is not None
        assert len(manifest["content_merkle_root"]) == 64

        # Verify records.jsonl content
        with open(export_dir / "records.jsonl") as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 3
        assert records[0]["id"] == hash1 or records[1]["id"] == hash1 or records[2]["id"] == hash1

    def test_import_cartridge(self, temp_dir):
        """Test importing a cartridge into a new cassette."""
        from memory_cassette import MemoryCassette

        # Create and export from first cassette
        db1_path = temp_dir / "cassette1.db"
        cassette1 = MemoryCassette(db_path=db1_path, agent_id="test_agent")

        hash1, _ = cassette1.memory_save("Memory one")
        hash2, _ = cassette1.memory_save("Memory two")

        export_dir = temp_dir / "export"
        manifest = cassette1.export_cartridge(export_dir)

        # Create new cassette and import
        db2_path = temp_dir / "cassette2.db"
        cassette2 = MemoryCassette(db_path=db2_path, agent_id="test_agent")

        result = cassette2.import_cartridge(export_dir)

        assert result["restored_records"] == 2
        assert result["merkle_verified"] is True

        # Verify memories are accessible
        recall1 = cassette2.memory_recall(hash1)
        recall2 = cassette2.memory_recall(hash2)

        assert recall1 is not None
        assert recall1["text"] == "Memory one"
        assert recall2 is not None
        assert recall2["text"] == "Memory two"

    def test_import_verifies_merkle_root(self, temp_dir):
        """Test that import verifies Merkle root."""
        from memory_cassette import MemoryCassette

        # Create and export
        db1_path = temp_dir / "cassette1.db"
        cassette1 = MemoryCassette(db_path=db1_path, agent_id="test_agent")
        cassette1.memory_save("Test memory")

        export_dir = temp_dir / "export"
        cassette1.export_cartridge(export_dir)

        # Corrupt the manifest
        manifest_path = export_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        manifest["content_merkle_root"] = "x" * 64  # Wrong root
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Try to import - should fail
        db2_path = temp_dir / "cassette2.db"
        cassette2 = MemoryCassette(db_path=db2_path, agent_id="test_agent")

        with pytest.raises(ValueError, match="Merkle root mismatch"):
            cassette2.import_cartridge(export_dir)


class TestCorruptAndRestore:
    """Test corrupt-and-restore functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests within project."""
        temp_path = get_test_dir()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_corrupt_and_restore(self, temp_dir):
        """Full corrupt-and-restore cycle."""
        from memory_cassette import MemoryCassette

        # Create cassette with memories
        db_path = temp_dir / "cassette.db"
        cassette = MemoryCassette(db_path=db_path, agent_id="test_agent")

        hash1, _ = cassette.memory_save("Important thought one")
        hash2, _ = cassette.memory_save("Important thought two")
        hash3, _ = cassette.memory_save("Important thought three")

        # Export cartridge before corruption
        export_dir = temp_dir / "backup"
        manifest = cassette.export_cartridge(export_dir)
        original_merkle = manifest["content_merkle_root"]

        # Simulate corruption by deleting the database
        os.remove(db_path)

        # Create new cassette and restore from cartridge
        cassette2 = MemoryCassette(db_path=db_path, agent_id="test_agent")
        result = cassette2.import_cartridge(export_dir)

        assert result["restored_records"] == 3
        assert result["merkle_verified"] is True

        # Verify all memories are restored
        assert cassette2.memory_recall(hash1) is not None
        assert cassette2.memory_recall(hash2) is not None
        assert cassette2.memory_recall(hash3) is not None

        # Verify Merkle root matches
        export_dir2 = temp_dir / "verify"
        manifest2 = cassette2.export_cartridge(export_dir2)
        assert manifest2["content_merkle_root"] == original_merkle

    def test_restore_preserves_content(self, temp_dir):
        """Test that restored content is byte-identical."""
        from memory_cassette import MemoryCassette

        original_text = "The exact content must be preserved!"

        # Create and export
        db1_path = temp_dir / "cassette1.db"
        cassette1 = MemoryCassette(db_path=db1_path, agent_id="test_agent")
        hash1, _ = cassette1.memory_save(original_text)

        export_dir = temp_dir / "export"
        cassette1.export_cartridge(export_dir)

        # Import to new cassette
        db2_path = temp_dir / "cassette2.db"
        cassette2 = MemoryCassette(db_path=db2_path, agent_id="test_agent")
        cassette2.import_cartridge(export_dir)

        # Verify content is identical
        recall = cassette2.memory_recall(hash1)
        assert recall["text"] == original_text


class TestReceiptChainIntegrity:
    """Test receipt chain integrity during operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests within project."""
        temp_path = get_test_dir()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_receipts_form_chain(self, temp_dir):
        """Test that receipts form a valid chain."""
        from memory_cassette import MemoryCassette
        from receipt_verifier import CassetteReceiptVerifier

        db_path = temp_dir / "cassette.db"
        cassette = MemoryCassette(db_path=db_path, agent_id="test_agent")

        # Save multiple memories
        cassette.memory_save("Memory 1")
        cassette.memory_save("Memory 2")
        cassette.memory_save("Memory 3")

        # Verify chain
        verifier = CassetteReceiptVerifier(db_path)
        result = verifier.verify_full_chain()

        assert result["valid"] is True
        assert result["chain_length"] == 3

    def test_session_merkle_root(self, temp_dir):
        """Test that session end computes Merkle root."""
        from memory_cassette import MemoryCassette

        db_path = temp_dir / "cassette.db"
        cassette = MemoryCassette(db_path=db_path, agent_id="test_agent")

        # Start session
        session = cassette.session_start("test_agent")
        session_id = session["session_id"]

        # Save memories in session
        cassette.memory_save("Session memory 1", session_id=session_id)
        cassette.memory_save("Session memory 2", session_id=session_id)

        # End session
        end_result = cassette.session_end(session_id)

        # Check Merkle root was computed
        assert "merkle_root" in end_result
        if end_result["merkle_root"]:
            assert len(end_result["merkle_root"]) == 64


class TestDeterminism:
    """Test determinism guarantees."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests within project."""
        temp_path = get_test_dir()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_content_hash_determinism(self, temp_dir):
        """Test that same content produces same hash."""
        from memory_cassette import MemoryCassette

        text = "Deterministic content test"

        # Create two cassettes
        db1_path = temp_dir / "cassette1.db"
        db2_path = temp_dir / "cassette2.db"

        cassette1 = MemoryCassette(db_path=db1_path, agent_id="test")
        cassette2 = MemoryCassette(db_path=db2_path, agent_id="test")

        hash1, _ = cassette1.memory_save(text)
        hash2, _ = cassette2.memory_save(text)

        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
