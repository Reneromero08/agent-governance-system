#!/usr/bin/env python3
"""
Phase 5.3.3 Tests: CODEBOOK_SYNC_PROTOCOL.md Validation

Tests that validate the Codebook Sync Protocol specification is complete,
well-formed, and defines all required protocol elements.

Deliverables verified:
    - CODEBOOK_SYNC_PROTOCOL.md exists and is normative
    - Sync handshake defined
    - Message shapes specified (SyncRequest, SyncResponse, SyncError)
    - Failure codes enumerated
    - Markov blanket semantics documented

Usage:
    pytest CAPABILITY/TESTBENCH/integration/test_phase_5_3_3_codebook_sync.py -v
"""

import hashlib
import re
import sys
from pathlib import Path

import pytest

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SYNC_PROTOCOL_PATH = PROJECT_ROOT / "LAW" / "CANON" / "SEMANTIC" / "CODEBOOK_SYNC_PROTOCOL.md"

sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def spec_content() -> str:
    """Load CODEBOOK_SYNC_PROTOCOL.md content."""
    assert SYNC_PROTOCOL_PATH.exists(), f"CODEBOOK_SYNC_PROTOCOL.md not found at {SYNC_PROTOCOL_PATH}"
    return SYNC_PROTOCOL_PATH.read_text(encoding='utf-8')


@pytest.fixture
def spec_hash(spec_content) -> str:
    """Compute SHA-256 of spec content."""
    return hashlib.sha256(spec_content.encode('utf-8')).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTENCE AND METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecExistence:
    """Test spec file exists and has proper metadata."""

    def test_spec_exists(self):
        """CODEBOOK_SYNC_PROTOCOL.md must exist."""
        assert SYNC_PROTOCOL_PATH.exists(), "CODEBOOK_SYNC_PROTOCOL.md not found"

    def test_spec_not_empty(self, spec_content):
        """Spec must have content."""
        assert len(spec_content) > 1000, "Spec appears too short"

    def test_spec_is_normative(self, spec_content):
        """Spec must be marked as NORMATIVE."""
        assert "Status:** NORMATIVE" in spec_content or "status: normative" in spec_content.lower()

    def test_spec_has_canon_id(self, spec_content):
        """Spec must have Canon ID."""
        assert "SEMANTIC-SYNC-001" in spec_content

    def test_spec_has_version(self, spec_content):
        """Spec must declare version."""
        assert re.search(r'\*\*Version:\*\*\s*\d+\.\d+\.\d+', spec_content)

    def test_spec_has_phase_reference(self, spec_content):
        """Spec must reference Phase 5.3.3."""
        assert "5.3.3" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC TUPLE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncTuple:
    """Test sync tuple is fully defined."""

    def test_sync_tuple_defined(self, spec_content):
        """Sync Tuple must be defined."""
        assert "SyncTuple" in spec_content or "sync_tuple" in spec_content

    def test_sync_tuple_fields(self, spec_content):
        """All sync tuple fields must be documented."""
        required_fields = [
            "codebook_id",
            "codebook_sha256",
            "codebook_semver",
            "kernel_version",
            "tokenizer_id",
        ]
        for field in required_fields:
            assert field in spec_content, f"Missing sync tuple field: {field}"


# ═══════════════════════════════════════════════════════════════════════════════
# HANDSHAKE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandshake:
    """Test handshake protocol is defined."""

    def test_handshake_section(self, spec_content):
        """Sync Handshake section must exist."""
        assert "## 2. Sync Handshake" in spec_content

    def test_handshake_flow(self, spec_content):
        """Handshake flow must be documented."""
        assert "SyncRequest" in spec_content
        assert "SyncResponse" in spec_content

    def test_handshake_states(self, spec_content):
        """Handshake states must be defined."""
        states = ["UNSYNCED", "PENDING", "SYNCED", "MISMATCHED", "FAILED"]
        for state in states:
            assert state in spec_content, f"Missing handshake state: {state}"

    def test_state_transitions(self, spec_content):
        """State transitions must be documented."""
        assert "State Transition" in spec_content or "──>" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE SHAPES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessageShapes:
    """Test message shapes are fully specified."""

    def test_message_shapes_section(self, spec_content):
        """Handshake Message Shapes section must exist."""
        assert "## 3. Handshake Message Shapes" in spec_content

    def test_sync_request_shape(self, spec_content):
        """SyncRequest shape must be defined."""
        assert "### 3.1 SyncRequest" in spec_content
        assert '"message_type": "SYNC_REQUEST"' in spec_content

    def test_sync_response_shape(self, spec_content):
        """SyncResponse shape must be defined."""
        assert "### 3.2 SyncResponse" in spec_content
        assert '"message_type": "SYNC_RESPONSE"' in spec_content

    def test_sync_error_shape(self, spec_content):
        """SyncError shape must be defined."""
        assert "### 3.3 SyncError" in spec_content
        assert '"message_type": "SYNC_ERROR"' in spec_content

    def test_match_response(self, spec_content):
        """Match response must be documented."""
        assert '"status": "MATCHED"' in spec_content

    def test_mismatch_response(self, spec_content):
        """Mismatch response must be documented."""
        assert '"status": "MISMATCHED"' in spec_content

    def test_blanket_status_field(self, spec_content):
        """blanket_status field must be present."""
        assert "blanket_status" in spec_content
        assert "ALIGNED" in spec_content
        assert "DISSOLVED" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE CODES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureCodes:
    """Test failure codes are enumerated."""

    def test_failure_codes_section(self, spec_content):
        """Failure Codes section must exist."""
        assert "## 4. Failure Codes" in spec_content

    def test_sync_specific_failures(self, spec_content):
        """Sync-specific failure codes must be defined."""
        sync_codes = [
            "E_SYNC_REQUIRED",
            "E_SYNC_EXPIRED",
            "E_SYNC_TIMEOUT",
            "E_PROTOCOL_VERSION",
            "E_BLANKET_DISSOLVED",
        ]
        for code in sync_codes:
            assert code in spec_content, f"Missing sync failure code: {code}"

    def test_codebook_failures(self, spec_content):
        """Codebook failure codes must be defined."""
        codebook_codes = [
            "E_CODEBOOK_MISMATCH",
            "E_KERNEL_VERSION",
            "E_TOKENIZER_MISMATCH",
        ]
        for code in codebook_codes:
            assert code in spec_content, f"Missing codebook failure code: {code}"

    def test_migration_failures(self, spec_content):
        """Migration failure codes must be defined."""
        migration_codes = [
            "E_MIGRATION_NOT_FOUND",
            "E_MIGRATION_FAILED",
        ]
        for code in migration_codes:
            assert code in spec_content, f"Missing migration failure code: {code}"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY POLICY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompatibilityPolicy:
    """Test compatibility policy is specified."""

    def test_compatibility_section(self, spec_content):
        """Compatibility Policy section must exist."""
        assert "## 5. Compatibility Policy" in spec_content

    def test_exact_match_policy(self, spec_content):
        """Exact match default policy must be documented."""
        assert "Exact Match" in spec_content or "exact match" in spec_content

    def test_semver_compatibility(self, spec_content):
        """Semver compatibility ranges must be documented."""
        assert "Semver" in spec_content or "semver" in spec_content

    def test_migration_protocol(self, spec_content):
        """Migration protocol must be documented."""
        assert "Migration Protocol" in spec_content or "migration_path" in spec_content

    def test_no_silent_migration(self, spec_content):
        """No silent migration rule must be stated."""
        assert "No Silent Migration" in spec_content or "NEVER silent" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CASSETTE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCassetteIntegration:
    """Test Cassette Network integration is documented."""

    def test_cassette_section(self, spec_content):
        """Cassette Network Integration section must exist."""
        assert "## 6. Cassette Network Integration" in spec_content

    def test_cassette_sync_state(self, spec_content):
        """Cassette sync state must be documented."""
        assert "Cassette Sync State" in spec_content or "SemanticCassette" in spec_content

    def test_verification_before_expansion(self, spec_content):
        """Verification before expansion must be documented."""
        assert "expand_pointer" in spec_content or "Verification Before Expansion" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# MARKOV BLANKET TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkovBlanket:
    """Test Markov blanket semantics are documented."""

    def test_markov_blanket_section(self, spec_content):
        """Markov Blanket Semantics section must exist."""
        assert "## 7. Markov Blanket Semantics" in spec_content

    def test_theoretical_foundation(self, spec_content):
        """Theoretical foundation must be documented."""
        assert "Theoretical Foundation" in spec_content or "Markov blanket" in spec_content

    def test_blanket_properties(self, spec_content):
        """Blanket properties must be documented."""
        assert "Conditional Independence" in spec_content or "P1:" in spec_content

    def test_active_inference(self, spec_content):
        """Active Inference interpretation must be documented."""
        assert "Active Inference" in spec_content

    def test_blanket_status_semantics(self, spec_content):
        """Blanket status semantics must be documented."""
        assert "ALIGNED" in spec_content
        assert "DISSOLVED" in spec_content
        assert "R-value" in spec_content or "R >" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# V1.1.0 EXTENSIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestV110Extensions:
    """Test v1.1.0 extensions are present."""

    def test_continuous_r_value(self, spec_content):
        """Continuous R-value section must exist."""
        assert "### 7.5 Continuous R-Value" in spec_content or "Continuous R-Value" in spec_content

    def test_m_field_interpretation(self, spec_content):
        """M Field Interpretation section must exist."""
        assert "### 7.6 M Field Interpretation" in spec_content or "M Field" in spec_content

    def test_blanket_health_tracking(self, spec_content):
        """Blanket Health Tracking section must exist."""
        assert "### 8.4 Blanket Health Tracking" in spec_content or "blanket_health" in spec_content

    def test_sigma_df_metric(self, spec_content):
        """σ^Df complexity metric must be documented."""
        assert "σ^Df" in spec_content or "sigma_df" in spec_content.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionManagement:
    """Test session management is documented."""

    def test_session_section(self, spec_content):
        """Session Management section must exist."""
        assert "## 8. Session Management" in spec_content

    def test_session_lifecycle(self, spec_content):
        """Session lifecycle must be documented."""
        assert "Session Lifecycle" in spec_content or "SessionInit" in spec_content

    def test_session_token(self, spec_content):
        """Session token must be documented."""
        assert "session_token" in spec_content
        assert "ttl" in spec_content.lower()

    def test_heartbeat(self, spec_content):
        """Heartbeat must be documented."""
        assert "SYNC_HEARTBEAT" in spec_content
        assert "HEARTBEAT_ACK" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# INFORMATION-THEORETIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestInformationTheoretic:
    """Test information-theoretic semantics are documented."""

    def test_info_theory_section(self, spec_content):
        """Information-Theoretic Semantics section must exist."""
        assert "## 10. Information-Theoretic Semantics" in spec_content

    def test_conditional_entropy(self, spec_content):
        """Conditional entropy must be documented."""
        assert "H(X|S)" in spec_content
        assert "conditional entropy" in spec_content.lower()

    def test_mutual_information(self, spec_content):
        """Mutual information must be documented."""
        assert "I(X;S)" in spec_content or "mutual information" in spec_content.lower()

    def test_semantic_density_connection(self, spec_content):
        """Semantic density connection must be documented."""
        assert "CDR" in spec_content
        assert "concept_units" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurity:
    """Test security considerations are documented."""

    def test_security_section(self, spec_content):
        """Security Considerations section must exist."""
        assert "## 11. Security Considerations" in spec_content

    def test_hash_collision_resistance(self, spec_content):
        """Hash collision resistance must be documented."""
        assert "collision" in spec_content.lower()
        assert "SHA-256" in spec_content

    def test_replay_protection(self, spec_content):
        """Replay protection must be documented."""
        assert "Replay" in spec_content or "timestamp" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-REFERENCES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossReferences:
    """Test internal cross-references are valid."""

    def test_references_spc_spec(self, spec_content):
        """Must reference SPC_SPEC.md."""
        assert "SPC_SPEC" in spec_content

    def test_references_gov_ir_spec(self, spec_content):
        """Must reference GOV_IR_SPEC.md."""
        assert "GOV_IR_SPEC" in spec_content

    def test_references_q33(self, spec_content):
        """Must reference Q33 (conditional entropy)."""
        assert "Q33" in spec_content or "q33" in spec_content

    def test_references_q35(self, spec_content):
        """Must reference Q35 (Markov blankets)."""
        assert "Q35" in spec_content or "q35" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDICES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAppendices:
    """Test appendices are present."""

    def test_message_type_registry(self, spec_content):
        """Message Type Registry appendix must exist."""
        assert "## Appendix A: Message Type Registry" in spec_content

    def test_sync_tuple_example(self, spec_content):
        """Sync Tuple example appendix must exist."""
        assert "## Appendix B: Canonical Sync Tuple" in spec_content

    def test_state_machine_diagram(self, spec_content):
        """State Machine Diagram appendix must exist."""
        assert "## Appendix C: State Machine" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT HASH RECEIPT
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentReceipt:
    """Test content integrity for receipt generation."""

    def test_content_hash_reproducible(self, spec_content, spec_hash):
        """Content hash must be reproducible."""
        content2 = SYNC_PROTOCOL_PATH.read_text(encoding='utf-8')
        hash2 = hashlib.sha256(content2.encode('utf-8')).hexdigest()
        assert spec_hash == hash2, "Content hash not reproducible"

    def test_content_hash_format(self, spec_hash):
        """Content hash must be valid SHA-256."""
        assert len(spec_hash) == 64
        assert all(c in '0123456789abcdef' for c in spec_hash)

    def test_changelog_present(self, spec_content):
        """Changelog section must be present."""
        assert "## Changelog" in spec_content

    def test_changelog_has_v110(self, spec_content):
        """Changelog must document v1.1.0."""
        assert "1.1.0" in spec_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
