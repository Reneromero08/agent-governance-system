"""
Session Capsule Tests (Phase 3.4 - Section A)

Tests for:
- A.1: Save/resume determinism (byte-identical replay)
- A.2: Partial execution resume (no state loss)
- A.3: Tamper detection (fail-closed on corruption)
- A.4: Hydration failure (fail-closed on unresolvable symbols)

Exit Criteria: All 4 fixtures green, determinism proven.
"""

import copy
import hashlib
import json
import sqlite3
import pytest
from pathlib import Path

from catalytic_chat.session_capsule import (
    SessionCapsule,
    SessionCapsuleError,
    _canonical_json,
    _compute_hash,
    EVENT_USER_MESSAGE,
    EVENT_ASSISTANT_RESPONSE,
    EVENT_TOOL_CALL,
    EVENT_EXPANSION,
)
from catalytic_chat.context_assembler import AssemblyReceipt


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def session_capsule(tmp_path):
    """Create a fresh SessionCapsule with isolated database."""
    db_path = tmp_path / "test_session.db"
    capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)
    yield capsule
    capsule.close()


@pytest.fixture
def make_assembly_receipt():
    """Factory for creating test AssemblyReceipt objects."""
    def _make(
        final_hash="abcd1234" * 8,
        tokens=100,
        working_set=None,
        pointer_set=None,
        corpus_snapshot_id="snap_001"
    ):
        return AssemblyReceipt(
            budget_used={"max_total_tokens": 1000},
            items_included=["sys", "u1"],
            items_excluded=[],
            final_assemblage_hash=final_hash,
            token_usage_total=tokens,
            success=True,
            working_set=working_set or ["@SYM1"],
            pointer_set=pointer_set or ["@SYM2"],
            corpus_snapshot_id=corpus_snapshot_id
        )
    return _make


# =============================================================================
# Helper Tests (Foundation)
# =============================================================================

class TestSessionCapsuleFoundation:
    """Foundational tests for SessionCapsule invariants."""

    def test_genesis_hash_constant(self):
        """Verify GENESIS_HASH is 64 zeros (SHA-256 null marker)."""
        assert SessionCapsule.GENESIS_HASH == "0" * 64
        assert len(SessionCapsule.GENESIS_HASH) == 64

    def test_canonical_json_determinism(self):
        """Verify _canonical_json produces identical output for identical input."""
        data = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}

        result1 = _canonical_json(data)
        result2 = _canonical_json(data)

        assert result1 == result2
        # Verify sorted keys
        assert result1 == b'{"a":2,"m":{"a":4,"b":3},"z":1}'

    def test_chain_hash_computation(self):
        """Verify chain_hash = SHA256(content_hash + prev_hash)."""
        content_hash = "a" * 64
        prev_hash = "b" * 64

        expected = hashlib.sha256(f"{content_hash}{prev_hash}".encode()).hexdigest()
        actual = _compute_hash(f"{content_hash}{prev_hash}".encode())

        assert actual == expected

    def test_events_append_only_trigger(self, session_capsule):
        """Verify SQLite trigger blocks UPDATE/DELETE on session_events."""
        session_id = session_capsule.create_session()
        session_capsule.log_user_message(session_id, "Test")

        conn = session_capsule._get_conn()

        # Attempt UPDATE - should be blocked by trigger
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("""
                UPDATE session_events SET payload_json = '{"hacked": true}'
                WHERE session_id = ?
            """, (session_id,))

        # Attempt DELETE - should be blocked by trigger
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("""
                DELETE FROM session_events WHERE session_id = ?
            """, (session_id,))


# =============================================================================
# A.1: Save/Resume Determinism Test
# =============================================================================

class TestSaveResumeDeterminism:
    """A.1: Verify save/resume produces byte-identical replay."""

    def test_save_resume_determinism_byte_identical(self, tmp_path):
        """
        A.1: Save a session, import to new DB, verify byte-identical export.

        Process:
        1. Create session with known events
        2. Export session
        3. Import to new database
        4. Export again
        5. Compare: exports must be byte-identical (excluding exported_at)
        """
        # Phase 1: Create original session
        db1_path = tmp_path / "original.db"
        capsule1 = SessionCapsule(db_path=db1_path, repo_root=tmp_path)

        session_id = capsule1.create_session(
            session_id="determinism_test",
            corpus_snapshot_id="corpus_v1"
        )

        # Add fixed events for determinism
        capsule1.log_user_message(session_id, "Hello world")
        capsule1.log_assistant_response(session_id, "Hello! How can I help?")
        capsule1.log_tool_call(session_id, "read_file", {"path": "/test.txt"})
        capsule1.log_tool_result(session_id, "read_file", {"content": "file contents"})

        # Export original
        export1 = capsule1.export_session(session_id)
        capsule1.close()

        # Phase 2: Import to fresh database
        db2_path = tmp_path / "imported.db"
        capsule2 = SessionCapsule(db_path=db2_path, repo_root=tmp_path)

        imported_session_id = capsule2.import_session(export1)

        # Export the imported session
        export2 = capsule2.export_session(imported_session_id)
        capsule2.close()

        # Phase 3: Verify byte-identical
        # Remove timestamps that vary (exported_at)
        export1_compare = {k: v for k, v in export1.items() if k != "exported_at"}
        export2_compare = {k: v for k, v in export2.items() if k != "exported_at"}

        # Canonical JSON comparison
        json1 = _canonical_json(export1_compare)
        json2 = _canonical_json(export2_compare)

        assert json1 == json2, "Exported sessions must be byte-identical"

        # Verify specific fields
        assert export1["session_id"] == export2["session_id"]
        assert export1["state"]["chain_head"] == export2["state"]["chain_head"]
        assert len(export1["events"]) == len(export2["events"])

        # Verify each event is identical
        for e1, e2 in zip(export1["events"], export2["events"]):
            assert e1["content_hash"] == e2["content_hash"]
            assert e1["prev_hash"] == e2["prev_hash"]
            assert e1["chain_hash"] == e2["chain_hash"]
            assert e1["payload"] == e2["payload"]

    def test_determinism_with_multiple_imports(self, tmp_path):
        """Verify multiple import cycles produce identical results."""
        # Create original
        db1_path = tmp_path / "original.db"
        capsule1 = SessionCapsule(db_path=db1_path, repo_root=tmp_path)

        session_id = capsule1.create_session(session_id="multi_import_test")
        capsule1.log_user_message(session_id, "Test message")
        export1 = capsule1.export_session(session_id)
        capsule1.close()

        # Import to DB2
        db2_path = tmp_path / "import2.db"
        capsule2 = SessionCapsule(db_path=db2_path, repo_root=tmp_path)
        capsule2.import_session(export1)
        export2 = capsule2.export_session(session_id)
        capsule2.close()

        # Import to DB3
        db3_path = tmp_path / "import3.db"
        capsule3 = SessionCapsule(db_path=db3_path, repo_root=tmp_path)
        capsule3.import_session(export2)
        export3 = capsule3.export_session(session_id)
        capsule3.close()

        # All chain_heads must match
        assert export1["state"]["chain_head"] == export2["state"]["chain_head"]
        assert export2["state"]["chain_head"] == export3["state"]["chain_head"]


# =============================================================================
# A.2: Partial Execution Resume Test
# =============================================================================

class TestPartialExecutionResume:
    """A.2: Verify partial execution resume preserves state."""

    def test_partial_execution_resume_no_state_loss(self, tmp_path, make_assembly_receipt):
        """
        A.2: Import session at event N, append more events, verify chain integrity.

        Process:
        1. Create session with N events including assembly (working/pointer sets)
        2. Export at event N
        3. Import to new database
        4. Append more events
        5. Verify chain integrity maintained
        6. Verify working/pointer sets preserved
        """
        # Phase 1: Create session with N events
        db1_path = tmp_path / "original.db"
        capsule1 = SessionCapsule(db_path=db1_path, repo_root=tmp_path)

        session_id = capsule1.create_session(
            session_id="resume_test",
            corpus_snapshot_id="corpus_v1"
        )

        # Initial events
        capsule1.log_user_message(session_id, "Query 1")
        capsule1.log_assistant_response(session_id, "Response 1")

        # Log assembly with working/pointer sets
        receipt = make_assembly_receipt(
            working_set=["@CANON/INVARIANTS", "@LAW/CONTRACT"],
            pointer_set=["@THOUGHT/LAB/NOTES"]
        )
        capsule1.log_assembly(session_id, receipt)

        # Capture state at N events
        state_at_n = capsule1.get_session_state(session_id)
        chain_head_at_n = state_at_n.chain_head
        event_count_at_n = state_at_n.event_count

        # Export
        export_data = capsule1.export_session(session_id)
        capsule1.close()

        # Phase 2: Import and resume
        db2_path = tmp_path / "resumed.db"
        capsule2 = SessionCapsule(db_path=db2_path, repo_root=tmp_path)

        imported_id = capsule2.import_session(export_data)

        # Verify state immediately after import
        state_after_import = capsule2.get_session_state(imported_id)
        assert state_after_import.chain_head == chain_head_at_n
        assert state_after_import.event_count == event_count_at_n
        assert set(state_after_import.working_set) == {"@CANON/INVARIANTS", "@LAW/CONTRACT"}
        assert set(state_after_import.pointer_set) == {"@THOUGHT/LAB/NOTES"}

        # Phase 3: Append more events
        capsule2.log_user_message(imported_id, "Query 2")
        capsule2.log_assistant_response(imported_id, "Response 2")

        # Phase 4: Verify chain integrity
        is_valid, error = capsule2.verify_chain(imported_id)
        assert is_valid, f"Chain integrity failed: {error}"

        # Verify new chain head links to old
        state_after_resume = capsule2.get_session_state(imported_id)
        assert state_after_resume.event_count == event_count_at_n + 2

        # Get events and verify chain links
        events = capsule2.get_events(imported_id)

        # Event N+1 should have prev_hash = chain_head_at_n
        event_n_plus_1 = events[event_count_at_n]  # 0-indexed
        assert event_n_plus_1.prev_hash == chain_head_at_n

        # Working/pointer sets still preserved
        assert set(state_after_resume.working_set) == {"@CANON/INVARIANTS", "@LAW/CONTRACT"}
        assert set(state_after_resume.pointer_set) == {"@THOUGHT/LAB/NOTES"}

        capsule2.close()

    def test_resume_preserves_corpus_snapshot_id(self, tmp_path):
        """Verify corpus_snapshot_id is preserved across resume."""
        db1_path = tmp_path / "original.db"
        capsule1 = SessionCapsule(db_path=db1_path, repo_root=tmp_path)

        session_id = capsule1.create_session(
            session_id="snapshot_test",
            corpus_snapshot_id="snapshot_abc123def456"
        )
        capsule1.log_user_message(session_id, "Test")
        export_data = capsule1.export_session(session_id)
        capsule1.close()

        db2_path = tmp_path / "resumed.db"
        capsule2 = SessionCapsule(db_path=db2_path, repo_root=tmp_path)
        capsule2.import_session(export_data)

        state = capsule2.get_session_state(session_id)
        assert state.corpus_snapshot_id == "snapshot_abc123def456"
        capsule2.close()


# =============================================================================
# A.3: Tamper Detection Tests
# =============================================================================

class TestTamperDetection:
    """A.3: Tamper detection tests for fail-closed behavior."""

    @pytest.fixture
    def valid_export(self, tmp_path):
        """Create a valid exported session for tampering tests."""
        db_path = tmp_path / "source.db"
        capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)

        session_id = capsule.create_session(session_id="tamper_test")
        capsule.log_user_message(session_id, "Test message")
        capsule.log_assistant_response(session_id, "Test response")

        export_data = capsule.export_session(session_id)
        capsule.close()
        return export_data

    def test_content_hash_corruption_detected(self, tmp_path, valid_export):
        """A.3a: Corrupted content_hash is detected."""
        # Corrupt content_hash of second event (index 1 = user_message after session_start)
        tampered = copy.deepcopy(valid_export)
        tampered["events"][1]["content_hash"] = "ff" * 32

        db_path = tmp_path / "target.db"
        capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)

        with pytest.raises(SessionCapsuleError) as exc_info:
            capsule.import_session(tampered)

        assert "content hash mismatch" in str(exc_info.value).lower()
        capsule.close()

    def test_prev_hash_corruption_detected(self, tmp_path, valid_export):
        """A.3b: Corrupted prev_hash is detected."""
        tampered = copy.deepcopy(valid_export)
        # Corrupt prev_hash to break chain linkage
        tampered["events"][1]["prev_hash"] = "ee" * 32

        db_path = tmp_path / "target.db"
        capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)

        with pytest.raises(SessionCapsuleError) as exc_info:
            capsule.import_session(tampered)

        assert "chain broken" in str(exc_info.value).lower()
        capsule.close()

    def test_chain_hash_corruption_detected(self, tmp_path, valid_export):
        """A.3c: Corrupted chain_hash is detected."""
        tampered = copy.deepcopy(valid_export)
        tampered["events"][1]["chain_hash"] = "dd" * 32

        db_path = tmp_path / "target.db"
        capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)

        with pytest.raises(SessionCapsuleError) as exc_info:
            capsule.import_session(tampered)

        assert "chain hash mismatch" in str(exc_info.value).lower()
        capsule.close()

    def test_payload_modification_detected(self, tmp_path, valid_export):
        """A.3d: Modified payload is detected via content_hash mismatch."""
        tampered = copy.deepcopy(valid_export)
        # Modify payload content without updating hash
        tampered["events"][1]["payload"]["content"] = "TAMPERED CONTENT"

        db_path = tmp_path / "target.db"
        capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)

        with pytest.raises(SessionCapsuleError) as exc_info:
            capsule.import_session(tampered)

        assert "content hash mismatch" in str(exc_info.value).lower()
        capsule.close()

    def test_import_fails_on_corruption(self, tmp_path, valid_export):
        """Verify import raises exception on corrupted data."""
        tampered = copy.deepcopy(valid_export)
        tampered["events"][1]["content_hash"] = "ff" * 32

        db_path = tmp_path / "target.db"
        capsule = SessionCapsule(db_path=db_path, repo_root=tmp_path)

        # Import should fail-closed with clear error
        with pytest.raises(SessionCapsuleError) as exc_info:
            capsule.import_session(tampered)

        # Error message should indicate the problem
        assert "invalid chain" in str(exc_info.value).lower() or \
               "content hash mismatch" in str(exc_info.value).lower()
        capsule.close()


# =============================================================================
# A.4: Hydration Failure Test
# =============================================================================

class TestHydrationFailure:
    """A.4: Verify fail-closed on unresolvable symbols during hydration."""

    def test_hydration_failure_fail_closed(self, tmp_path):
        """
        A.4: Verify fail-closed when symbols cannot be resolved.

        Process:
        1. Create session with expansion event referencing nonexistent symbol
        2. Export and import to new DB
        3. Create CortexExpansionResolver in empty environment
        4. Attempt to resolve symbol
        5. Verify CortexRetrievalError raised
        """
        from catalytic_chat.cortex_expansion_resolver import (
            CortexExpansionResolver,
            CortexRetrievalError
        )

        # Phase 1: Create session with symbol references
        db1_path = tmp_path / "source.db"
        capsule1 = SessionCapsule(db_path=db1_path, repo_root=tmp_path)

        session_id = capsule1.create_session(session_id="hydration_test")
        capsule1.log_user_message(session_id, "Query about @NONEXISTENT_SYMBOL")

        # Log expansion event that references a symbol
        capsule1.log_expansion(
            session_id,
            symbol_id="@NONEXISTENT_SYMBOL",
            content_hash="abc123" * 10 + "abcd",
            source="cortex"
        )

        export_data = capsule1.export_session(session_id)
        capsule1.close()

        # Phase 2: Import to clean environment
        db2_path = tmp_path / "target.db"
        capsule2 = SessionCapsule(db_path=db2_path, repo_root=tmp_path)

        # Import succeeds (data is valid)
        imported_id = capsule2.import_session(export_data)

        # Phase 3: Attempt to resolve symbol via CORTEX
        # Create resolver in isolated environment (no actual CORTEX)
        resolver = CortexExpansionResolver(
            repo_root=tmp_path,  # Empty repo, no symbols
            fail_on_unresolved=True
        )

        # Get the expansion event
        events = capsule2.get_events(imported_id, event_type=EVENT_EXPANSION)
        assert len(events) > 0
        expansion_event = events[0]
        symbol_id = expansion_event.payload["symbol_id"]

        # Attempt resolution - should fail closed
        with pytest.raises(CortexRetrievalError) as exc_info:
            resolver.resolve_expansion(symbol_id)

        assert "failed to resolve" in str(exc_info.value).lower()
        assert symbol_id in str(exc_info.value)

        capsule2.close()

    def test_hydration_batch_fail_closed(self, tmp_path):
        """A.4 variant: Batch resolution fails closed on first unresolvable."""
        from catalytic_chat.cortex_expansion_resolver import (
            CortexExpansionResolver,
            CortexRetrievalError
        )

        resolver = CortexExpansionResolver(
            repo_root=tmp_path,
            fail_on_unresolved=True
        )

        symbols = ["@MISSING_1", "@MISSING_2", "@MISSING_3"]

        with pytest.raises(CortexRetrievalError):
            resolver.resolve_batch(symbols, is_explicit=True)

    def test_hydration_graceful_mode_returns_empty(self, tmp_path):
        """A.4 variant: When fail_on_unresolved=False, returns empty content."""
        from catalytic_chat.cortex_expansion_resolver import CortexExpansionResolver

        resolver = CortexExpansionResolver(
            repo_root=tmp_path,
            fail_on_unresolved=False  # Graceful mode
        )

        result = resolver.resolve_expansion("@NONEXISTENT")

        assert result.content == ""
        assert result.source == "unresolved"
        # Verify resolution was attempted through the current retrieval chain
        assert any(step in result.retrieval_path for step in [
            "spc_resolve", "cassette_network_symbol", "cassette_network",
            "symbol_registry", "docs_index_fts"
        ])

    def test_hydration_stats_track_failures(self, tmp_path):
        """Verify resolver stats track failure count."""
        from catalytic_chat.cortex_expansion_resolver import CortexExpansionResolver

        resolver = CortexExpansionResolver(
            repo_root=tmp_path,
            fail_on_unresolved=False
        )

        # Resolve multiple missing symbols
        resolver.resolve_expansion("@MISSING_1")
        resolver.resolve_expansion("@MISSING_2")

        stats = resolver.get_stats()
        assert stats["failures"] >= 2
        assert stats["total_queries"] >= 2
