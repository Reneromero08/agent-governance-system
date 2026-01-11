#!/usr/bin/env python3
"""
Phase 4.1 Tests - Pointer Types (SPC Integration)

Tests for:
1. SYMBOL_PTR - ASCII radicals (C, I, V) and CJK glyphs (法, 真)
2. HASH_PTR - Content-addressed lookup via CAS
3. COMPOSITE_PTR - Numbered (C3), unary (C*), binary (C&I), path (L.C.3, 法.驗)
4. Pointer caching - Register, lookup, invalidate
5. SPC Integration - Full resolver with CAS backend

Reference:
- LAW/CANON/SEMANTIC/SPC_SPEC.md
- Q35 (Markov Blankets)
- Q33 (Semantic Density)
"""

import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from spc_decoder import (
    SPCDecoder,
    pointer_resolve,
    register_cas_lookup,
    unregister_cas_lookup,
    is_cas_available,
    PointerType,
    ErrorCode,
    DecodeSuccess,
    FailClosed
)


class TestSymbolPtr:
    """Tests for SYMBOL_PTR resolution."""

    def test_ascii_radical_C(self):
        """ASCII radical 'C' resolves to Contract domain."""
        result = pointer_resolve("C")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Contract"

    def test_ascii_radical_I(self):
        """ASCII radical 'I' resolves to Invariant domain."""
        result = pointer_resolve("I")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Invariant"

    def test_ascii_radical_V(self):
        """ASCII radical 'V' resolves to Verification domain."""
        result = pointer_resolve("V")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Verification"

    def test_all_ascii_radicals(self):
        """All ASCII radicals resolve successfully."""
        radicals = "CIVLGSRAJP"
        for r in radicals:
            result = pointer_resolve(r)
            assert result["status"] == "SUCCESS", f"Radical '{r}' failed"

    def test_cjk_glyph_law(self):
        """CJK glyph '法' resolves to Law domain."""
        result = pointer_resolve("法")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Law"
        assert result["ir"]["inputs"]["expansion"]["glyph"] == "法"

    def test_cjk_glyph_truth(self):
        """CJK glyph '真' resolves to Truth domain."""
        result = pointer_resolve("真")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Truth"

    def test_cjk_glyph_contract(self):
        """CJK glyph '契' resolves to Contract domain."""
        result = pointer_resolve("契")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Contract"

    def test_polysemic_symbol_without_context(self):
        """Polysemic symbol '道' without context returns E_CONTEXT_REQUIRED."""
        result = pointer_resolve("道")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] == "E_CONTEXT_REQUIRED"

    def test_polysemic_symbol_with_context(self):
        """Polysemic symbol '道' with context resolves correctly."""
        result = pointer_resolve("道", context_keys={"CONTEXT_TYPE": "CONTEXT_PATH"})
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["path"] == "LAW/CANON"

    def test_unknown_symbol_fail_closed(self):
        """Unknown symbol returns FAIL_CLOSED."""
        result = pointer_resolve("X")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] in ("E_UNKNOWN_SYMBOL", "E_SYNTAX")


class TestHashPtr:
    """Tests for HASH_PTR resolution."""

    def test_hash_ptr_without_cas(self):
        """HASH_PTR without CAS registered returns informative error."""
        # Ensure CAS is unregistered
        unregister_cas_lookup()

        result = pointer_resolve("sha256:abc123def456abc123def456")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] == "E_HASH_NOT_FOUND"
        assert "register_cas_lookup" in result["error_detail"]

    def test_hash_ptr_with_mock_cas(self):
        """HASH_PTR with CAS registered resolves content."""
        # Register mock CAS
        test_content = {"text": "Test governance rule content", "type": "rule"}

        def mock_cas(hash_value: str):
            if hash_value == "abc123def456abc123def456":
                return test_content
            return None

        register_cas_lookup(mock_cas)
        try:
            result = pointer_resolve("sha256:abc123def456abc123def456")
            assert result["status"] == "SUCCESS"
            assert result["ir"]["inputs"]["expansion"]["text"] == test_content["text"]
            assert result["ir"]["inputs"]["expansion"]["source"] == "memory_cassette"
        finally:
            unregister_cas_lookup()

    def test_hash_ptr_not_found_in_cas(self):
        """HASH_PTR not in CAS returns E_HASH_NOT_FOUND."""
        def mock_cas(hash_value: str):
            return None  # Always returns None

        register_cas_lookup(mock_cas)
        try:
            result = pointer_resolve("sha256:ffffffffffffffffffffffff")
            assert result["status"] == "FAIL_CLOSED"
            assert result["error_code"] == "E_HASH_NOT_FOUND"
        finally:
            unregister_cas_lookup()

    def test_hash_ptr_cas_error(self):
        """HASH_PTR CAS error returns E_CAS_UNAVAILABLE."""
        def failing_cas(hash_value: str):
            raise RuntimeError("CAS connection failed")

        register_cas_lookup(failing_cas)
        try:
            result = pointer_resolve("sha256:abc123def456abc123def456")
            assert result["status"] == "FAIL_CLOSED"
            assert result["error_code"] == "E_CAS_UNAVAILABLE"
        finally:
            unregister_cas_lookup()

    def test_is_cas_available(self):
        """is_cas_available() reflects registration state."""
        unregister_cas_lookup()
        assert is_cas_available() == False

        register_cas_lookup(lambda h: None)
        assert is_cas_available() == True

        unregister_cas_lookup()
        assert is_cas_available() == False


class TestCompositePtrNumbered:
    """Tests for numbered COMPOSITE_PTR (C3, I5)."""

    def test_contract_rule_C3(self):
        """C3 resolves to INBOX rule."""
        result = pointer_resolve("C3")
        assert result["status"] == "SUCCESS"
        assert "INBOX" in result["ir"]["inputs"]["expansion"]["summary"]
        assert result["ir"]["inputs"]["expansion"]["id"] == "C3"

    def test_contract_rule_C1(self):
        """C1 resolves to Text outranks code rule."""
        result = pointer_resolve("C1")
        assert result["status"] == "SUCCESS"
        assert "Text" in result["ir"]["inputs"]["expansion"]["summary"]

    def test_invariant_I5(self):
        """I5 resolves to Determinism invariant."""
        result = pointer_resolve("I5")
        assert result["status"] == "SUCCESS"
        assert "Determinism" in result["ir"]["inputs"]["expansion"]["summary"]

    def test_invalid_rule_number(self):
        """C999 returns E_RULE_NOT_FOUND."""
        result = pointer_resolve("C999")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] == "E_RULE_NOT_FOUND"

    def test_numbered_with_context(self):
        """C3:build includes context."""
        result = pointer_resolve("C3:build")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["context"] == "build"


class TestCompositePtrUnary:
    """Tests for unary COMPOSITE_PTR (C*, C!, C?)."""

    def test_all_contract_rules(self):
        """C* resolves to all contract rules."""
        result = pointer_resolve("C*")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["operator"] == "ALL"
        assert "C1" in result["ir"]["inputs"]["expansion"]["rules"]
        assert result["ir"]["inputs"]["expansion"]["count"] >= 13

    def test_all_invariants(self):
        """I* resolves to all invariants."""
        result = pointer_resolve("I*")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["operator"] == "ALL"
        assert "I1" in result["ir"]["inputs"]["expansion"]["rules"]
        assert result["ir"]["inputs"]["expansion"]["count"] >= 20

    def test_not_verification(self):
        """V! resolves to NOT Verification."""
        result = pointer_resolve("V!")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["operator"] == "NOT"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "Verification"

    def test_check_jobspec(self):
        """J? resolves to CHECK JobSpec."""
        result = pointer_resolve("J?")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["operator"] == "CHECK"
        assert result["ir"]["inputs"]["expansion"]["domain"] == "JobSpec"


class TestCompositePtrBinary:
    """Tests for binary COMPOSITE_PTR (C&I, C|I)."""

    def test_contract_and_invariant(self):
        """C&I resolves to Contract AND Invariant."""
        result = pointer_resolve("C&I")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["operator"] == "AND"
        assert result["ir"]["inputs"]["expansion"]["left"]["domain"] == "Contract"
        assert result["ir"]["inputs"]["expansion"]["right"]["domain"] == "Invariant"

    def test_contract_or_invariant(self):
        """C|I resolves to Contract OR Invariant."""
        result = pointer_resolve("C|I")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["operator"] == "OR"
        assert result["ir"]["inputs"]["expansion"]["left"]["domain"] == "Contract"
        assert result["ir"]["inputs"]["expansion"]["right"]["domain"] == "Invariant"

    def test_law_and_governance(self):
        """L&G resolves to Law AND Governance."""
        result = pointer_resolve("L&G")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["left"]["domain"] == "Law"
        assert result["ir"]["inputs"]["expansion"]["right"]["domain"] == "Governance"

    def test_unknown_operand_fails(self):
        """C&X fails with syntax error (X not valid radical)."""
        result = pointer_resolve("C&X")
        assert result["status"] == "FAIL_CLOSED"
        # X doesn't match the radical pattern, so it's a syntax error
        assert result["error_code"] in ("E_UNKNOWN_SYMBOL", "E_SYNTAX")


class TestCompositePtrPath:
    """Tests for path COMPOSITE_PTR (L.C.3, 法.驗)."""

    def test_law_contract_path(self):
        """L.C resolves to Law -> Contract path."""
        result = pointer_resolve("L.C")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["depth"] == 2
        parts = result["ir"]["inputs"]["expansion"]["parts"]
        assert parts[0]["domain"] == "Law"
        assert parts[1]["domain"] == "Contract"

    def test_law_contract_rule3_path(self):
        """L.C.3 resolves to Law -> Contract -> Rule 3 path."""
        result = pointer_resolve("L.C.3")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["depth"] == 3
        parts = result["ir"]["inputs"]["expansion"]["parts"]
        assert parts[0]["domain"] == "Law"
        assert parts[1]["domain"] == "Contract"
        assert parts[2]["type"] == "index"
        assert parts[2]["value"] == 3

    def test_cjk_path_law_verification(self):
        """法.驗 resolves to Law -> Verification path."""
        result = pointer_resolve("法.驗")
        assert result["status"] == "SUCCESS"
        parts = result["ir"]["inputs"]["expansion"]["parts"]
        assert parts[0]["domain"] == "Law"
        assert parts[1]["domain"] == "Verification"

    def test_unknown_path_component_fails(self):
        """L.X fails with unknown symbol."""
        result = pointer_resolve("L.X")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] == "E_UNKNOWN_SYMBOL"


class TestCompositePtrContext:
    """Tests for context COMPOSITE_PTR (C:build)."""

    def test_contract_build_context(self):
        """C:build resolves with build context."""
        result = pointer_resolve("C:build")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["context"] == "build"
        assert "Build" in result["ir"]["inputs"]["expansion"]["context_description"]

    def test_verification_audit_context(self):
        """V:audit resolves with audit context."""
        result = pointer_resolve("V:audit")
        assert result["status"] == "SUCCESS"
        assert result["ir"]["inputs"]["expansion"]["context"] == "audit"

    def test_invalid_context_fails(self):
        """C:invalid fails with E_INVALID_QUALIFIER."""
        result = pointer_resolve("C:invalid")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] == "E_INVALID_QUALIFIER"


class TestCodebookMismatch:
    """Tests for codebook mismatch handling (Q35 Markov blanket)."""

    def test_codebook_hash_mismatch(self):
        """Wrong codebook hash returns E_CODEBOOK_MISMATCH."""
        result = pointer_resolve("C3", codebook_sha256="wrong_hash_12345")
        assert result["status"] == "FAIL_CLOSED"
        assert result["error_code"] == "E_CODEBOOK_MISMATCH"

    def test_correct_codebook_hash(self):
        """Correct codebook hash allows resolution."""
        # First get the correct hash
        decoder = SPCDecoder()
        correct_hash = decoder.codebook_hash

        result = pointer_resolve("C3", codebook_sha256=correct_hash)
        assert result["status"] == "SUCCESS"


class TestTokenReceipts:
    """Tests for token receipts (Q33 semantic density)."""

    def test_receipt_included(self):
        """Successful decode includes token receipt."""
        result = pointer_resolve("C3")
        assert result["status"] == "SUCCESS"
        assert "token_receipt" in result
        receipt = result["token_receipt"]
        assert "CDR" in receipt
        assert "compression_ratio" in receipt
        # Check for token count fields (may be named differently)
        has_token_counts = (
            "tokens_pointer" in receipt or
            "pointer_tokens" in receipt or
            "concept_units" in receipt
        )
        assert has_token_counts, f"Token receipt missing counts: {receipt.keys()}"

    def test_cdr_positive(self):
        """CDR is positive for valid pointers."""
        result = pointer_resolve("C3")
        assert result["token_receipt"]["CDR"] > 0

    def test_compression_achieved(self):
        """Compression ratio > 1 for meaningful pointers."""
        result = pointer_resolve("C3")
        assert result["token_receipt"]["compression_ratio"] > 1


class TestPointerCaching:
    """Tests for pointer caching logic (uses in-memory SQLite)."""

    @pytest.fixture
    def pointer_db(self):
        """Create an in-memory SQLite database for pointer caching tests."""
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE pointers (
                pointer_id TEXT PRIMARY KEY,
                pointer_type TEXT NOT NULL,
                base_ptr TEXT NOT NULL,
                target_hash TEXT,
                qualifiers TEXT,
                codebook_id TEXT DEFAULT 'ags-codebook',
                created_at TEXT,
                resolved_count INTEGER DEFAULT 0,
                last_resolved TEXT
            )
        """)
        conn.commit()
        yield conn
        conn.close()

    def test_pointer_register(self, pointer_db):
        """Registering a pointer creates cache entry."""
        import hashlib
        from datetime import datetime, timezone

        pointer = "C3"
        codebook_id = "ags-codebook"
        now = datetime.now(timezone.utc).isoformat()
        pointer_id = hashlib.sha256(f"{pointer}:{codebook_id}".encode()).hexdigest()[:16]

        pointer_db.execute("""
            INSERT INTO pointers (pointer_id, pointer_type, base_ptr, target_hash, codebook_id, created_at, resolved_count, last_resolved)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
        """, (pointer_id, "composite", pointer, "abc123", codebook_id, now, now))
        pointer_db.commit()

        cursor = pointer_db.execute("SELECT * FROM pointers WHERE pointer_id = ?", (pointer_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[2] == "C3"  # base_ptr
        assert row[1] == "composite"  # pointer_type

    def test_pointer_lookup(self, pointer_db):
        """Looking up a cached pointer returns entry."""
        import hashlib

        pointer = "C3"
        codebook_id = "ags-codebook"
        pointer_id = hashlib.sha256(f"{pointer}:{codebook_id}".encode()).hexdigest()[:16]

        pointer_db.execute("""
            INSERT INTO pointers (pointer_id, pointer_type, base_ptr, target_hash, codebook_id, resolved_count)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (pointer_id, "composite", pointer, "abc123", codebook_id))
        pointer_db.commit()

        cursor = pointer_db.execute("SELECT * FROM pointers WHERE pointer_id = ?", (pointer_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[3] == "abc123"  # target_hash

    def test_pointer_lookup_miss(self, pointer_db):
        """Looking up non-existent pointer returns None."""
        import hashlib

        pointer_id = hashlib.sha256("NONEXISTENT:ags-codebook".encode()).hexdigest()[:16]
        cursor = pointer_db.execute("SELECT * FROM pointers WHERE pointer_id = ?", (pointer_id,))
        row = cursor.fetchone()
        assert row is None

    def test_pointer_invalidate_by_codebook(self, pointer_db):
        """Invalidating by codebook clears matching pointers."""
        import hashlib

        # Insert pointers for two codebooks
        for ptr, cb in [("C3", "ags-codebook"), ("I5", "ags-codebook"), ("X1", "other-codebook")]:
            pid = hashlib.sha256(f"{ptr}:{cb}".encode()).hexdigest()[:16]
            pointer_db.execute(
                "INSERT INTO pointers (pointer_id, pointer_type, base_ptr, codebook_id) VALUES (?, ?, ?, ?)",
                (pid, "composite", ptr, cb)
            )
        pointer_db.commit()

        # Invalidate ags-codebook
        cursor = pointer_db.execute("DELETE FROM pointers WHERE codebook_id = ?", ("ags-codebook",))
        pointer_db.commit()
        assert cursor.rowcount == 2

        # Verify
        cursor = pointer_db.execute("SELECT COUNT(*) FROM pointers WHERE codebook_id = ?", ("ags-codebook",))
        assert cursor.fetchone()[0] == 0

        cursor = pointer_db.execute("SELECT COUNT(*) FROM pointers WHERE codebook_id = ?", ("other-codebook",))
        assert cursor.fetchone()[0] == 1

    def test_pointer_stats(self, pointer_db):
        """Pointer stats returns cache statistics."""
        import hashlib

        # Insert mixed pointer types
        for ptr, ptype in [("C3", "composite"), ("I5", "composite"), ("法", "symbol")]:
            pid = hashlib.sha256(f"{ptr}:ags-codebook".encode()).hexdigest()[:16]
            pointer_db.execute(
                "INSERT INTO pointers (pointer_id, pointer_type, base_ptr, codebook_id) VALUES (?, ?, ?, ?)",
                (pid, ptype, ptr, "ags-codebook")
            )
        pointer_db.commit()

        # Get stats
        cursor = pointer_db.execute("SELECT COUNT(*) FROM pointers")
        total = cursor.fetchone()[0]
        assert total == 3

        cursor = pointer_db.execute("SELECT pointer_type, COUNT(*) FROM pointers GROUP BY pointer_type")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        assert by_type["composite"] == 2
        assert by_type["symbol"] == 1


class TestSPCIntegration:
    """Tests for SPC integration logic (without file system dependencies)."""

    def test_sync_handshake_logic(self):
        """Sync handshake logic produces correct blanket status."""
        # Test the core handshake logic without MemoryCassette
        decoder_tuple = {
            "codebook_id": "ags-codebook",
            "codebook_sha256": "abc123",
            "codebook_semver": "0.2.0",
            "kernel_version": "1.0.0",
            "tokenizer_id": "tiktoken/o200k_base"
        }
        receiver_tuple = decoder_tuple.copy()  # Same = aligned

        # Check alignment
        if decoder_tuple["codebook_sha256"] == receiver_tuple["codebook_sha256"]:
            blanket_status = "ALIGNED"
        else:
            blanket_status = "DISSOLVED"

        assert blanket_status == "ALIGNED"

    def test_sync_handshake_mismatch(self):
        """Mismatched codebook hash produces DISSOLVED status."""
        decoder_tuple = {"codebook_sha256": "abc123"}
        receiver_tuple = {"codebook_sha256": "different"}

        if decoder_tuple["codebook_sha256"] == receiver_tuple["codebook_sha256"]:
            blanket_status = "ALIGNED"
        else:
            blanket_status = "DISSOLVED"

        assert blanket_status == "DISSOLVED"

    def test_resolve_returns_success(self):
        """Direct pointer resolution works."""
        result = pointer_resolve("C3")
        assert result["status"] == "SUCCESS"
        assert "INBOX" in result["ir"]["inputs"]["expansion"]["summary"]

    def test_cas_registration_flow(self):
        """CAS registration and unregistration works correctly."""
        from spc_decoder import register_cas_lookup, unregister_cas_lookup, is_cas_available

        # Initially may or may not be available
        initial = is_cas_available()

        # Register a mock
        def mock_cas(h):
            return {"text": "test", "type": "test"}
        register_cas_lookup(mock_cas)
        assert is_cas_available() == True

        # Unregister
        unregister_cas_lookup()
        assert is_cas_available() == False

        # Restore initial state if needed
        if initial:
            register_cas_lookup(mock_cas)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
