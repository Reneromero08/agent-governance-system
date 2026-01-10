#!/usr/bin/env python3
"""
Phase 5.2.7 Token Accountability Tests

Integration tests for the Token Accountability system:
- TokenReceipt schema validation
- Semantic search receipt emission
- Session aggregation
- Firewall rule enforcement

Exit Criteria:
- Every semantic_query emits TokenReceipt
- Session summaries show cumulative savings
- Firewall rejects unreceipted large outputs
- 10+ tests passing
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Fix path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class TestTokenReceiptSchema(unittest.TestCase):
    """Tests for TokenReceipt schema and dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        from CAPABILITY.PRIMITIVES.token_receipt import (
            TokenReceipt, TokenizerInfo, QueryMetadata,
            get_default_tokenizer, validate_receipt, SCHEMA_VERSION
        )
        self.TokenReceipt = TokenReceipt
        self.TokenizerInfo = TokenizerInfo
        self.QueryMetadata = QueryMetadata
        self.get_default_tokenizer = get_default_tokenizer
        self.validate_receipt = validate_receipt
        self.SCHEMA_VERSION = SCHEMA_VERSION

    def test_required_fields(self):
        """Test that required fields are enforced."""
        tokenizer = self.get_default_tokenizer()
        receipt = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=100,
            tokenizer=tokenizer,
        )
        data = receipt.to_dict()

        # Check required fields present
        self.assertEqual(data["schema_version"], self.SCHEMA_VERSION)
        self.assertEqual(data["operation"], "semantic_query")
        self.assertEqual(data["tokens_out"], 100)
        self.assertIn("tokenizer", data)
        self.assertEqual(data["tokenizer"]["library"], tokenizer.library)

    def test_auto_computed_fields(self):
        """Test that tokens_saved and savings_pct are auto-computed."""
        tokenizer = self.get_default_tokenizer()
        receipt = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=100,
            tokenizer=tokenizer,
            baseline_equiv=1000,
        )

        self.assertEqual(receipt.tokens_saved, 900)
        self.assertAlmostEqual(receipt.savings_pct, 90.0, places=2)

    def test_receipt_hash_determinism(self):
        """Test that receipt_hash is deterministic for same inputs."""
        tokenizer = self.TokenizerInfo(
            library="tiktoken",
            encoding="o200k_base",
            version="0.12.0",
        )

        # Create two receipts with exact same data (including timestamp)
        # to verify hash computation is deterministic
        receipt1 = self.TokenReceipt(
            operation="scl_decode",
            tokens_out=10,
            tokenizer=tokenizer,
            baseline_equiv=100,
            timestamp_utc="2026-01-01T00:00:00Z",
        )
        receipt2 = self.TokenReceipt(
            operation="scl_decode",
            tokens_out=10,
            tokenizer=tokenizer,
            baseline_equiv=100,
            timestamp_utc="2026-01-01T00:00:00Z",  # Same timestamp
        )

        # Same inputs -> same receipt_hash
        self.assertEqual(receipt1.receipt_hash, receipt2.receipt_hash)

        # Also verify hash is 64 hex chars (SHA-256)
        self.assertEqual(len(receipt1.receipt_hash), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in receipt1.receipt_hash))

    def test_operation_id_unique(self):
        """Test that operation_id differs for different timestamps."""
        tokenizer = self.get_default_tokenizer()

        # Use explicit different timestamps to ensure uniqueness
        receipt1 = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=100,
            tokenizer=tokenizer,
            timestamp_utc="2026-01-01T00:00:00.000001Z",
        )
        receipt2 = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=100,
            tokenizer=tokenizer,
            timestamp_utc="2026-01-01T00:00:00.000002Z",
        )

        # operation_id includes timestamp, so should be different
        self.assertNotEqual(receipt1.operation_id, receipt2.operation_id)

    def test_invalid_operation_rejected(self):
        """Test that invalid operations are rejected."""
        tokenizer = self.get_default_tokenizer()

        with self.assertRaises(ValueError) as ctx:
            self.TokenReceipt(
                operation="invalid_operation",
                tokens_out=100,
                tokenizer=tokenizer,
            )

        self.assertIn("Invalid operation", str(ctx.exception))

    def test_validate_receipt_function(self):
        """Test the validate_receipt function."""
        tokenizer = self.get_default_tokenizer()
        receipt = self.TokenReceipt(
            operation="cas_get",
            tokens_out=50,
            tokenizer=tokenizer,
            baseline_equiv=200,
        )

        # Should validate successfully
        self.assertTrue(self.validate_receipt(receipt.to_dict()))

    def test_display_formats(self):
        """Test compact and verbose display formats."""
        tokenizer = self.get_default_tokenizer()
        receipt = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=834,
            tokenizer=tokenizer,
            baseline_equiv=624170,
        )

        compact = receipt.compact()
        self.assertIn("[TOKEN]", compact)
        self.assertIn("semantic_query", compact)
        self.assertIn("834", compact)
        self.assertIn("99.", compact)  # High savings percentage

        verbose = receipt.verbose()
        self.assertIn("TOKEN RECEIPT", verbose)
        self.assertIn("Operation:", verbose)
        self.assertIn("Tokens Out:", verbose)


class TestSessionAggregator(unittest.TestCase):
    """Tests for TokenSession aggregation."""

    def setUp(self):
        """Set up test fixtures."""
        from CAPABILITY.PRIMITIVES.token_receipt import (
            TokenReceipt, get_default_tokenizer
        )
        from CAPABILITY.PRIMITIVES.token_session import (
            TokenSession, SessionTokenSummary,
            start_new_session, end_current_session, log_receipt
        )
        self.TokenReceipt = TokenReceipt
        self.get_default_tokenizer = get_default_tokenizer
        self.TokenSession = TokenSession
        self.SessionTokenSummary = SessionTokenSummary
        self.start_new_session = start_new_session
        self.end_current_session = end_current_session
        self.log_receipt = log_receipt

    def test_session_aggregation(self):
        """Test that session correctly aggregates receipts."""
        tokenizer = self.get_default_tokenizer()
        session = self.TokenSession(session_id="test-agg-001")

        receipts = [
            self.TokenReceipt(
                operation="semantic_query",
                tokens_out=100,
                tokenizer=tokenizer,
                tokens_in=10,
                baseline_equiv=1000,
            ),
            self.TokenReceipt(
                operation="scl_decode",
                tokens_out=5,
                tokenizer=tokenizer,
                tokens_in=5,
                baseline_equiv=50,
            ),
        ]

        for r in receipts:
            session.add_receipt(r)

        summary = session.get_summary()

        self.assertEqual(summary.receipts_count, 2)
        self.assertEqual(summary.total_tokens_in, 15)
        self.assertEqual(summary.total_tokens_out, 105)
        self.assertEqual(summary.total_baseline_equiv, 1050)
        self.assertEqual(summary.total_tokens_saved, 945)

    def test_operations_breakdown(self):
        """Test that operations are counted correctly."""
        tokenizer = self.get_default_tokenizer()
        session = self.TokenSession(session_id="test-ops-001")

        for _ in range(3):
            session.add_receipt(self.TokenReceipt(
                operation="semantic_query",
                tokens_out=100,
                tokenizer=tokenizer,
            ))

        session.add_receipt(self.TokenReceipt(
            operation="scl_decode",
            tokens_out=10,
            tokenizer=tokenizer,
        ))

        summary = session.get_summary()
        self.assertEqual(summary.operations_breakdown["semantic_query"], 3)
        self.assertEqual(summary.operations_breakdown["scl_decode"], 1)

    def test_session_end(self):
        """Test that ending session sets ended_at."""
        session = self.TokenSession(session_id="test-end-001")
        self.assertIsNone(session.ended_at)

        summary = session.end_session()
        self.assertIsNotNone(session.ended_at)
        self.assertIsNotNone(summary.ended_at)

    def test_global_session_management(self):
        """Test global session start/end functions."""
        session = self.start_new_session("test-global-001")
        self.assertEqual(session.session_id, "test-global-001")

        summary = self.end_current_session()
        self.assertIsNotNone(summary)
        self.assertEqual(summary.session_id, "test-global-001")


class TestFirewallEnforcement(unittest.TestCase):
    """Tests for TokenFirewall rule enforcement."""

    def setUp(self):
        """Set up test fixtures."""
        from CAPABILITY.PRIMITIVES.token_receipt import (
            TokenReceipt, get_default_tokenizer
        )
        from CAPABILITY.PRIMITIVES.token_firewall import (
            TokenFirewall, FirewallVerdict,
            validate_token_output, require_token_receipt
        )
        from CAPABILITY.PRIMITIVES.token_session import start_new_session

        self.TokenReceipt = TokenReceipt
        self.get_default_tokenizer = get_default_tokenizer
        self.TokenFirewall = TokenFirewall
        self.FirewallVerdict = FirewallVerdict
        self.validate_token_output = validate_token_output
        self.require_token_receipt = require_token_receipt
        self.start_new_session = start_new_session

        # Start fresh session for each test
        self.start_new_session("test-firewall")

    def test_reject_large_output_without_receipt(self):
        """Test REJECT-001: Large outputs without receipt are rejected."""
        firewall = self.TokenFirewall()

        result = firewall.validate_output(1500, receipt=None)

        self.assertEqual(result.verdict, self.FirewallVerdict.REJECT)
        self.assertFalse(result.allowed)
        self.assertEqual(result.rule_id, "REJECT-001")

    def test_allow_small_output_without_receipt(self):
        """Test that small outputs without receipt are allowed."""
        firewall = self.TokenFirewall()

        result = firewall.validate_output(500, receipt=None)

        self.assertEqual(result.verdict, self.FirewallVerdict.ALLOW)
        self.assertTrue(result.allowed)

    def test_allow_large_output_with_receipt(self):
        """Test that large outputs with receipt are allowed."""
        firewall = self.TokenFirewall(auto_log_receipts=False)
        tokenizer = self.get_default_tokenizer()

        receipt = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=1500,
            tokenizer=tokenizer,
            baseline_equiv=624170,
        )

        result = firewall.validate_output(1500, receipt=receipt)

        self.assertEqual(result.verdict, self.FirewallVerdict.ALLOW)
        self.assertTrue(result.allowed)

    def test_warn_low_savings(self):
        """Test WARN-001: Low savings triggers warning."""
        firewall = self.TokenFirewall(auto_log_receipts=False)
        tokenizer = self.get_default_tokenizer()

        receipt = self.TokenReceipt(
            operation="semantic_query",
            tokens_out=400,
            tokenizer=tokenizer,
            baseline_equiv=500,  # Only 20% savings
        )

        result = firewall.validate_output(400, receipt=receipt)

        self.assertEqual(result.verdict, self.FirewallVerdict.WARN)
        self.assertTrue(result.allowed)  # Warning doesn't block
        self.assertIn("WARN-001", result.warnings[0])

    def test_require_receipt_raises(self):
        """Test that require_receipt raises for large unreceipted outputs."""
        firewall = self.TokenFirewall()

        with self.assertRaises(ValueError) as ctx:
            firewall.require_receipt(2000, None, "test context")

        self.assertIn("TokenFirewall", str(ctx.exception))
        self.assertIn("test context", str(ctx.exception))

    def test_violation_logging(self):
        """Test that violations are logged."""
        firewall = self.TokenFirewall()

        firewall.validate_output(1500, receipt=None)
        firewall.validate_output(2000, receipt=None)

        violations = firewall.get_violations()
        self.assertEqual(len(violations), 2)
        self.assertEqual(violations[0]["rule_id"], "REJECT-001")


class TestSemanticSearchReceipt(unittest.TestCase):
    """Tests for semantic search receipt emission."""

    def test_search_response_has_receipt(self):
        """Test that SearchResponse includes TokenReceipt."""
        # Import and check structure
        from NAVIGATION.CORTEX.semantic.semantic_search import SearchResponse

        # SearchResponse should have receipt field
        self.assertTrue(hasattr(SearchResponse, '__dataclass_fields__'))
        self.assertIn('receipt', SearchResponse.__dataclass_fields__)

    def test_search_response_iterable(self):
        """Test that SearchResponse is backwards-compatible (iterable)."""
        from NAVIGATION.CORTEX.semantic.semantic_search import (
            SearchResponse, SearchResult
        )

        results = [
            SearchResult(hash="abc", content="test", similarity=0.9),
            SearchResult(hash="def", content="test2", similarity=0.8),
        ]
        response = SearchResponse(results=results, receipt=None)

        # Should be iterable
        self.assertEqual(len(response), 2)
        self.assertEqual(response[0].hash, "abc")

        # Should work in for loop
        count = 0
        for r in response:
            count += 1
        self.assertEqual(count, 2)


class TestJobSpecIntegration(unittest.TestCase):
    """Tests for JobSpec token_receipt field."""

    def test_jobspec_schema_has_token_receipt(self):
        """Test that JobSpec schema includes token_receipt field."""
        schema_path = PROJECT_ROOT / "LAW" / "SCHEMAS" / "jobspec.schema.json"
        self.assertTrue(schema_path.exists())

        with open(schema_path) as f:
            schema = json.load(f)

        self.assertIn("token_receipt", schema["properties"])

    def test_scl_decode_emits_receipt(self):
        """Test that SCL decode includes token_receipt."""
        try:
            from CAPABILITY.TOOLS.scl.scl_cli import decode_program

            # Decode a simple program
            result = decode_program("C3", emit_token_receipt=True)

            if result.get("ok"):
                # Should have token_receipt in response
                self.assertIn("token_receipt", result)
                self.assertIn("token_receipt", result["jobspec"])
        except ImportError:
            self.skipTest("SCL CLI not available")


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
