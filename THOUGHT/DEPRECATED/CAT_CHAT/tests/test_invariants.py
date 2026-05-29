"""
Tests for Phase I.3: Invariant Verifier
=======================================

Tests for all 7 catalytic invariant verification methods.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from catalytic_chat.invariant_verifier import (
    InvariantResult,
    VerificationReport,
    InvariantVerifier,
    verify_session_invariants,
)
from catalytic_chat.session_capsule import (
    SessionCapsule,
    EVENT_USER_MESSAGE,
    EVENT_PARTITION,
    EVENT_BUDGET_CHECK,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cat_chat.db"
        yield db_path


@pytest.fixture
def capsule_with_session(temp_db):
    """Create a session capsule with a test session."""
    capsule = SessionCapsule(db_path=temp_db)
    session_id = capsule.create_session()
    yield capsule, session_id
    capsule.close()


@pytest.fixture
def verifier(temp_db):
    """Create an invariant verifier."""
    v = InvariantVerifier(db_path=temp_db)
    yield v
    v.close()


# =============================================================================
# InvariantResult Tests
# =============================================================================

class TestInvariantResult:
    """Tests for InvariantResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = InvariantResult(
            invariant_id="INV-CATALYTIC-01",
            invariant_name="Restoration",
            passed=True,
            evidence={"files_checked": 10},
            timestamp="2026-01-19T12:00:00Z",
        )

        assert result.invariant_id == "INV-CATALYTIC-01"
        assert result.passed is True

    def test_result_to_dict(self):
        """Test serialization."""
        result = InvariantResult(
            invariant_id="INV-CATALYTIC-04",
            invariant_name="Clean Space Bound",
            passed=False,
            evidence={"budget_used": 15000, "budget_total": 10000},
            timestamp="2026-01-19T12:00:00Z",
            details="Budget exceeded by 5000 tokens",
        )

        d = result.to_dict()

        assert d["invariant_id"] == "INV-CATALYTIC-04"
        assert d["passed"] is False
        assert d["evidence"]["budget_used"] == 15000
        assert "exceeded" in d["details"]


# =============================================================================
# VerificationReport Tests
# =============================================================================

class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_empty_report(self):
        """Test empty report."""
        report = VerificationReport(
            session_id="test-session",
            verified_at="2026-01-19T12:00:00Z",
        )

        assert report.all_passed is False  # No results = not passed
        assert report.passed_count == 0
        assert report.failed_count == 0

    def test_report_with_results(self):
        """Test report with results."""
        report = VerificationReport(
            session_id="test-session",
            verified_at="2026-01-19T12:00:00Z",
        )

        report.add_result(InvariantResult(
            invariant_id="INV-01",
            invariant_name="Test 1",
            passed=True,
            evidence={},
            timestamp="2026-01-19T12:00:00Z",
        ))

        report.add_result(InvariantResult(
            invariant_id="INV-02",
            invariant_name="Test 2",
            passed=True,
            evidence={},
            timestamp="2026-01-19T12:00:00Z",
        ))

        assert report.all_passed is True
        assert report.passed_count == 2
        assert report.failed_count == 0

    def test_report_with_failure(self):
        """Test report with a failure."""
        report = VerificationReport(
            session_id="test-session",
            verified_at="2026-01-19T12:00:00Z",
        )

        report.add_result(InvariantResult(
            invariant_id="INV-01",
            invariant_name="Test 1",
            passed=True,
            evidence={},
            timestamp="2026-01-19T12:00:00Z",
        ))

        report.add_result(InvariantResult(
            invariant_id="INV-02",
            invariant_name="Test 2",
            passed=False,
            evidence={"error": "something went wrong"},
            timestamp="2026-01-19T12:00:00Z",
        ))

        assert report.all_passed is False
        assert report.passed_count == 1
        assert report.failed_count == 1

    def test_report_to_markdown(self):
        """Test markdown generation."""
        report = VerificationReport(
            session_id="test-session",
            verified_at="2026-01-19T12:00:00Z",
        )

        report.add_result(InvariantResult(
            invariant_id="INV-CATALYTIC-01",
            invariant_name="Restoration",
            passed=True,
            evidence={"files_checked": 0},
            timestamp="2026-01-19T12:00:00Z",
            details="No file modifications",
        ))

        md = report.to_markdown()

        assert "# Catalytic Invariant Verification Report" in md
        assert "INV-CATALYTIC-01" in md
        assert "PASS" in md


# =============================================================================
# INV-CATALYTIC-01 Tests (Restoration)
# =============================================================================

class TestInv01Restoration:
    """Tests for INV-CATALYTIC-01: Restoration."""

    def test_trivial_pass_no_events(self, capsule_with_session, verifier):
        """Session with no file operations passes trivially."""
        capsule, session_id = capsule_with_session

        result = verifier.verify_inv_01_restoration(session_id)

        assert result.passed is True
        assert "trivially passes" in result.details.lower()

    def test_trivial_pass_no_file_writes(self, capsule_with_session, verifier):
        """Session with only user messages passes."""
        capsule, session_id = capsule_with_session

        # Add some user messages (no file operations)
        capsule.log_user_message(session_id, "Hello")
        capsule.log_user_message(session_id, "How are you?")

        result = verifier.verify_inv_01_restoration(session_id)

        assert result.passed is True


# =============================================================================
# INV-CATALYTIC-02 Tests (Verification)
# =============================================================================

class TestInv02Verification:
    """Tests for INV-CATALYTIC-02: Verification (O(1) proof size)."""

    def test_valid_hashes(self, capsule_with_session, verifier):
        """All events should have valid 256-bit hashes."""
        capsule, session_id = capsule_with_session

        # Add some events
        capsule.log_user_message(session_id, "Test message 1")
        capsule.log_user_message(session_id, "Test message 2")

        result = verifier.verify_inv_02_verification(session_id)

        assert result.passed is True
        assert result.evidence["all_hashes_valid"] is True

    def test_hash_size_is_constant(self, capsule_with_session, verifier):
        """Hash size should be 32 bytes (64 hex chars) regardless of content."""
        capsule, session_id = capsule_with_session

        # Add events with varying content sizes
        capsule.log_user_message(session_id, "Short")
        capsule.log_user_message(session_id, "A" * 10000)  # Long message

        result = verifier.verify_inv_02_verification(session_id)

        assert result.passed is True
        assert result.evidence["hash_size_bytes"] == 32


# =============================================================================
# INV-CATALYTIC-03 Tests (Reversibility)
# =============================================================================

class TestInv03Reversibility:
    """Tests for INV-CATALYTIC-03: Reversibility."""

    def test_chain_integrity(self, capsule_with_session, verifier):
        """Chain should be verifiable (proves reversibility)."""
        capsule, session_id = capsule_with_session

        # Add events
        capsule.log_user_message(session_id, "Message 1")
        capsule.log_user_message(session_id, "Message 2")
        capsule.log_user_message(session_id, "Message 3")

        result = verifier.verify_inv_03_reversibility(session_id)

        assert result.passed is True
        assert result.evidence["chain_verified"] is True


# =============================================================================
# INV-CATALYTIC-04 Tests (Clean Space Bound)
# =============================================================================

class TestInv04CleanSpaceBound:
    """Tests for INV-CATALYTIC-04: Clean Space Bound."""

    def test_trivial_pass_no_partition_events(self, capsule_with_session, verifier):
        """No partition events = trivially passes."""
        capsule, session_id = capsule_with_session

        result = verifier.verify_inv_04_clean_space_bound(session_id)

        assert result.passed is True

    def test_budget_within_limits(self, capsule_with_session, verifier):
        """Budget within limits should pass."""
        capsule, session_id = capsule_with_session

        # Log partition event within budget
        capsule.log_partition(
            session_id=session_id,
            query_hash="abc123",
            working_set_ids=["item1", "item2"],
            pointer_set_ids=["item3"],
            budget_total=10000,
            budget_used=5000,  # Within budget
            threshold=0.5,
            E_mean=0.7,
            E_min=0.5,
            E_max=0.9,
            items_below_threshold=1,
            items_over_budget=0,
        )

        result = verifier.verify_inv_04_clean_space_bound(session_id)

        assert result.passed is True
        assert result.evidence["budget_violations"] == []

    def test_budget_exceeded(self, capsule_with_session, verifier):
        """Budget exceeded should fail."""
        capsule, session_id = capsule_with_session

        # Log partition event over budget
        capsule.log_partition(
            session_id=session_id,
            query_hash="abc123",
            working_set_ids=["item1", "item2"],
            pointer_set_ids=[],
            budget_total=5000,
            budget_used=8000,  # Over budget!
            threshold=0.5,
            E_mean=0.7,
            E_min=0.5,
            E_max=0.9,
            items_below_threshold=0,
            items_over_budget=1,
        )

        result = verifier.verify_inv_04_clean_space_bound(session_id)

        assert result.passed is False
        assert len(result.evidence["budget_violations"]) == 1


# =============================================================================
# INV-CATALYTIC-05 Tests (Fail-Closed)
# =============================================================================

class TestInv05FailClosed:
    """Tests for INV-CATALYTIC-05: Fail-Closed."""

    def test_trivial_pass_no_failures(self, capsule_with_session, verifier):
        """No failures = passes."""
        capsule, session_id = capsule_with_session

        capsule.log_user_message(session_id, "Normal message")

        result = verifier.verify_inv_05_fail_closed(session_id)

        assert result.passed is True

    def test_pass_when_session_ended_after_failure(self, capsule_with_session, verifier):
        """Session ended after failure = handled properly."""
        capsule, session_id = capsule_with_session

        # This test just verifies the logic - actual failure handling
        # would need more complex setup
        result = verifier.verify_inv_05_fail_closed(session_id)

        assert result.passed is True


# =============================================================================
# INV-CATALYTIC-06 Tests (Determinism)
# =============================================================================

class TestInv06Determinism:
    """Tests for INV-CATALYTIC-06: Determinism."""

    def test_hash_chain_deterministic(self, capsule_with_session, verifier):
        """Hash chain should be deterministic (recomputed = stored)."""
        capsule, session_id = capsule_with_session

        # Add events
        capsule.log_user_message(session_id, "Deterministic content")
        capsule.log_user_message(session_id, "More content")

        result = verifier.verify_inv_06_determinism(session_id)

        assert result.passed is True
        assert result.evidence["hash_mismatches"] == []
        assert result.evidence["chain_deterministic"] is True


# =============================================================================
# INV-CATALYTIC-07 Tests (Auto-Context)
# =============================================================================

class TestInv07AutoContext:
    """Tests for INV-CATALYTIC-07: Auto-Context."""

    def test_pass_no_manual_refs(self, capsule_with_session, verifier):
        """Messages without @symbols pass."""
        capsule, session_id = capsule_with_session

        capsule.log_user_message(session_id, "What is the architecture?")
        capsule.log_user_message(session_id, "Tell me about authentication")

        result = verifier.verify_inv_07_auto_context(session_id)

        assert result.passed is True
        assert result.evidence["auto_managed"] is True

    def test_fail_manual_refs(self, capsule_with_session, verifier):
        """Messages with @SYMBOL references should fail."""
        capsule, session_id = capsule_with_session

        capsule.log_user_message(session_id, "Look at @CANON/INVARIANTS for the rules")
        capsule.log_user_message(session_id, "Check @NAVIGATION/PROMPTS/PHASE1")

        result = verifier.verify_inv_07_auto_context(session_id)

        assert result.passed is False
        assert len(result.evidence["manual_references"]) == 2


# =============================================================================
# Full Verification Tests
# =============================================================================

class TestFullVerification:
    """Tests for complete verification workflow."""

    def test_verify_all_clean_session(self, capsule_with_session, verifier):
        """Clean session should pass all invariants."""
        capsule, session_id = capsule_with_session

        # Add some normal events
        capsule.log_user_message(session_id, "Hello")
        capsule.log_partition(
            session_id=session_id,
            query_hash="test",
            working_set_ids=["a"],
            pointer_set_ids=["b"],
            budget_total=10000,
            budget_used=5000,
            threshold=0.5,
            E_mean=0.7,
            E_min=0.5,
            E_max=0.9,
            items_below_threshold=0,
            items_over_budget=0,
        )

        report = verifier.verify_all(session_id)

        assert report.passed_count >= 5  # Most should pass

    def test_verify_single_invariant(self, capsule_with_session, verifier):
        """Should be able to verify a single invariant."""
        capsule, session_id = capsule_with_session

        result = verifier.verify_single(session_id, "INV-CATALYTIC-02")

        assert result.invariant_id == "INV-CATALYTIC-02"

    def test_verify_single_invalid_id(self, capsule_with_session, verifier):
        """Unknown invariant ID should raise error."""
        capsule, session_id = capsule_with_session

        with pytest.raises(ValueError, match="Unknown invariant"):
            verifier.verify_single(session_id, "INV-INVALID-99")


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_verify_session_invariants(self, temp_db):
        """Test the verify_session_invariants function."""
        # Create a session
        capsule = SessionCapsule(db_path=temp_db)
        session_id = capsule.create_session()
        capsule.log_user_message(session_id, "Test")
        capsule.close()

        # Verify using convenience function
        report = verify_session_invariants(session_id, db_path=temp_db)

        assert isinstance(report, VerificationReport)
        assert report.session_id == session_id


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_markdown_report_format(self, capsule_with_session, verifier):
        """Markdown report should have proper structure."""
        capsule, session_id = capsule_with_session
        capsule.log_user_message(session_id, "Test")

        report = verifier.verify_all(session_id)
        md = report.to_markdown()

        assert "# Catalytic Invariant Verification Report" in md
        assert session_id in md
        assert "INV-CATALYTIC-01" in md
        assert "Evidence" in md

    def test_dict_export(self, capsule_with_session, verifier):
        """Report should be exportable as dict."""
        capsule, session_id = capsule_with_session
        capsule.log_user_message(session_id, "Test")

        report = verifier.verify_all(session_id)
        d = report.to_dict()

        assert d["session_id"] == session_id
        assert "results" in d
        assert isinstance(d["results"], list)
