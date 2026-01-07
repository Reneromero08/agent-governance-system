#!/usr/bin/env python3
"""
Integration Tests for Phase 2.4.2: Protected Artifact Inventory

Tests the protected artifacts inventory and scanner primitives.

Exit Criteria:
- Protected roots/patterns explicitly defined and machine-readable
- Scanner detects protected artifacts deterministically
- Scanner fails-closed if protected artifacts appear in public pack modes
- Inventory completeness verified
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.protected_inventory import (
    ProtectedInventory,
    ProtectedPattern,
    ArtifactClass,
    DistributionPolicy,
    get_default_inventory
)
from CAPABILITY.PRIMITIVES.protected_scanner import (
    ProtectedScanner,
    ScanResult
)


class TestProtectedInventory:
    """Tests for ProtectedInventory class."""

    def test_inventory_creation(self):
        """Test creating an inventory with patterns."""
        inventory = ProtectedInventory()

        pattern = ProtectedPattern(
            artifact_class=ArtifactClass.VECTOR_DATABASE,
            patterns=["**/*.db"],
            allowed_locations=["data/**"],
            distribution_policy=DistributionPolicy.PLAINTEXT_NEVER,
            description="Test vector database"
        )

        inventory.add_pattern(pattern)

        assert len(inventory.patterns) == 1
        assert inventory.patterns[0].artifact_class == ArtifactClass.VECTOR_DATABASE

    def test_inventory_determinism(self):
        """Test that inventory hashing is deterministic."""
        inv1 = get_default_inventory()
        inv2 = get_default_inventory()

        assert inv1.hash() == inv2.hash()

    def test_inventory_serialization(self):
        """Test inventory JSON serialization round-trip."""
        original = get_default_inventory()
        json_str = original.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert "version" in parsed
        assert "patterns" in parsed

        # Round-trip
        loaded = ProtectedInventory.from_json(json_str)
        assert loaded.hash() == original.hash()

    def test_path_matching(self):
        """Test pattern matching against paths."""
        inventory = get_default_inventory()

        # Test vector database match
        db_path = Path("NAVIGATION/CORTEX/db/system1.db")
        assert inventory.is_protected(db_path)

        matches = inventory.find_matching_patterns(db_path)
        assert len(matches) > 0
        assert matches[0].artifact_class == ArtifactClass.VECTOR_DATABASE

        # Test non-protected path
        code_path = Path("CAPABILITY/PRIMITIVES/cas_store.py")
        assert not inventory.is_protected(code_path)

    def test_sealing_requirements(self):
        """Test sealing requirement logic for different contexts."""
        inventory = get_default_inventory()

        # PLAINTEXT_NEVER artifact
        vector_db = Path("NAVIGATION/CORTEX/db/system1.db")
        assert inventory.requires_sealing(vector_db, context="public")
        assert inventory.requires_sealing(vector_db, context="internal")
        assert inventory.requires_sealing(vector_db, context="working")

        # PLAINTEXT_INTERNAL artifact
        proof_manifest = Path("NAVIGATION/PROOFS/GREEN_STATE.json")
        assert inventory.requires_sealing(proof_manifest, context="public")
        assert not inventory.requires_sealing(proof_manifest, context="internal")


class TestProtectedScanner:
    """Tests for ProtectedScanner with fixtures."""

    @pytest.fixture
    def test_repo(self, tmp_path):
        """Create a test repository with protected artifacts."""
        repo = tmp_path / "test_repo"
        repo.mkdir()

        # Create vector database (protected)
        (repo / "NAVIGATION").mkdir(parents=True)
        (repo / "NAVIGATION" / "CORTEX").mkdir(parents=True)
        (repo / "NAVIGATION" / "CORTEX" / "db").mkdir(parents=True)
        vector_db = repo / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
        vector_db.write_bytes(b"fake vector database content")

        # Create compression proof (protected)
        (repo / "NAVIGATION" / "PROOFS").mkdir(parents=True)
        (repo / "NAVIGATION" / "PROOFS" / "COMPRESSION").mkdir(parents=True)
        compression_proof = repo / "NAVIGATION" / "PROOFS" / "COMPRESSION" / "COMPRESSION_PROOF_DATA.json"
        compression_proof.write_text('{"test": "data"}')

        # Create public file (not protected)
        readme = repo / "README.md"
        readme.write_text("# Test Repository")

        return repo

    def test_scanner_detects_protected_artifacts(self, test_repo):
        """Test that scanner detects protected artifacts."""
        scanner = ProtectedScanner(repo_root=test_repo)
        result = scanner.scan_directory(context="working")

        assert result.verdict in ["PASS", "WARN"]
        assert result.protected_count >= 2
        assert result.total_files_scanned >= 3

        # Verify specific matches
        paths = [str(m.path) for m in result.matches]
        assert any("system1.db" in p for p in paths)
        assert any("COMPRESSION_PROOF_DATA.json" in p for p in paths)

    def test_scanner_fails_in_public_context(self, test_repo):
        """Test that scanner fails-closed in public context."""
        scanner = ProtectedScanner(repo_root=test_repo)
        result = scanner.scan_directory(context="public")

        assert result.verdict == "FAIL"
        assert len(result.violations) > 0
        assert result.protected_count >= 2

    def test_scanner_passes_with_no_protected_artifacts(self, tmp_path):
        """Test that scanner passes when no protected artifacts exist."""
        repo = tmp_path / "clean_repo"
        repo.mkdir()

        # Create only public files
        (repo / "README.md").write_text("# Clean Repo")
        (repo / "src").mkdir()
        (repo / "src" / "main.py").write_text("print('hello')")

        scanner = ProtectedScanner(repo_root=repo)
        result = scanner.scan_directory(context="public")

        assert result.verdict == "PASS"
        assert result.protected_count == 0
        assert len(result.violations) == 0

    def test_scanner_determinism(self, test_repo):
        """Test that scanner produces deterministic results."""
        scanner = ProtectedScanner(repo_root=test_repo)

        result1 = scanner.scan_directory(context="working")
        result2 = scanner.scan_directory(context="working")

        assert result1.protected_count == result2.protected_count
        assert result1.verdict == result2.verdict

        # Verify matches are in the same order
        paths1 = [str(m.path) for m in result1.matches]
        paths2 = [str(m.path) for m in result2.matches]
        assert paths1 == paths2

    def test_scanner_receipt_format(self, test_repo):
        """Test that scanner produces valid JSON receipts."""
        scanner = ProtectedScanner(repo_root=test_repo)
        result = scanner.scan_directory(context="working", compute_hashes=True)

        # Serialize to JSON
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert "verdict" in parsed
        assert "context" in parsed
        assert "matches" in parsed
        assert "violations" in parsed
        assert "inventory_hash" in parsed
        assert "scan_timestamp" in parsed

        # Verify matches have expected fields
        if parsed["matches"]:
            match = parsed["matches"][0]
            assert "path" in match
            assert "artifact_class" in match
            assert "distribution_policy" in match
            assert "size_bytes" in match

    def test_file_list_scanning(self, test_repo):
        """Test scanning a specific list of files."""
        scanner = ProtectedScanner(repo_root=test_repo)

        files = [
            Path("NAVIGATION/CORTEX/db/system1.db"),
            Path("README.md")
        ]

        result = scanner.scan_file_list(files, context="public")

        assert result.verdict == "FAIL"  # Contains protected artifact
        assert result.protected_count >= 1
        assert result.total_files_scanned == 2


class TestInventoryCompleteness:
    """Tests to verify inventory completeness."""

    def test_all_vector_databases_covered(self):
        """Verify all known vector database patterns are in inventory."""
        inventory = get_default_inventory()

        test_paths = [
            "NAVIGATION/CORTEX/db/system1.db",
            "NAVIGATION/CORTEX/_generated/cortex.db",
            "THOUGHT/LAB/CAT_CHAT/CAT_CORTEX/db/system1.db",
        ]

        for path in test_paths:
            assert inventory.is_protected(Path(path)), f"Missing coverage for {path}"

    def test_all_proof_outputs_covered(self):
        """Verify all proof output patterns are in inventory."""
        inventory = get_default_inventory()

        test_paths = [
            "NAVIGATION/PROOFS/GREEN_STATE.json",
            "NAVIGATION/PROOFS/PROOF_MANIFEST.json",
            "NAVIGATION/PROOFS/COMPRESSION/COMPRESSION_PROOF_DATA.json",
        ]

        for path in test_paths:
            assert inventory.is_protected(Path(path)), f"Missing coverage for {path}"

    def test_cas_blobs_covered(self):
        """Verify CAS blob patterns are in inventory."""
        inventory = get_default_inventory()

        test_paths = [
            ".ags-cas/objects/abc123",
            ".ags-cas/refs/main",
        ]

        for path in test_paths:
            assert inventory.is_protected(Path(path)), f"Missing coverage for {path}"


class TestFailureScenarios:
    """Tests for expected failure modes."""

    def test_missing_protected_artifact_in_public_pack(self, tmp_path):
        """Test that missing sealing triggers FAIL verdict."""
        repo = tmp_path / "public_pack"
        repo.mkdir()

        # Simulate a pack output with protected artifact
        (repo / "NAVIGATION" / "CORTEX" / "db").mkdir(parents=True)
        (repo / "NAVIGATION" / "CORTEX" / "db" / "vectors.db").write_bytes(b"vectors")

        scanner = ProtectedScanner(repo_root=repo)
        result = scanner.scan_directory(context="public")

        assert result.verdict == "FAIL"
        assert "PLAINTEXT_NEVER" in str(result.violations)

    def test_tampered_inventory_detected(self):
        """Test that inventory changes are detected via hash."""
        inv1 = get_default_inventory()
        hash1 = inv1.hash()

        inv2 = get_default_inventory()
        inv2.patterns.pop()  # Remove one pattern
        hash2 = inv2.hash()

        assert hash1 != hash2


if __name__ == "__main__":
    # Run with: python -m pytest test_phase_2_4_2_protected_inventory.py -v
    pytest.main([__file__, "-v", "--tb=short"])
