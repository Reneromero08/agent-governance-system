---
uuid: f8c3d2a1-9b7e-4f5c-a6d8-2e1b3c5d7f9a
title: "Security Tests Investigation - Cassette Network Adversarial Testing Analysis"
section: report
bucket: capability/testbench
author: Claude Opus 4.5
priority: High
created: 2026-01-25
modified: 2026-01-25
status: Complete
summary: "Full investigation of xfailed adversarial/security tests in test_negative_controls.py. Analysis reveals these tests are testing EMBEDDING MODEL BEHAVIOR (vocabulary overlap), not actual security boundaries. The real security is enforced elsewhere via fail-closed validation, capability constraints, and trust boundaries."
tags:
- security-testing
- cassette-network
- adversarial-testing
- embedding-model
- investigation
---

# Security Tests Investigation - Cassette Network Adversarial Testing

**Date:** 2026-01-25
**Investigator:** Claude Opus 4.5
**Status:** COMPLETE

---

## Executive Summary

The xfailed "security" tests in `test_negative_controls.py` are NOT security tests at all. They are **embedding model quality tests** that happen to use security-themed query strings. The tests fail because semantic embedding models (all-MiniLM-L6-v2) match vocabulary, not intent - a fundamental property of how embeddings work, not a security vulnerability.

**Key Finding:** The cassette network's actual security guarantees are enforced at different layers:
1. **Trust Boundaries** (ADR-026): Routers/models are untrusted; all outputs validated
2. **Fail-Closed Validation**: Schema validation, capability constraints, hash verification
3. **Capability System**: Revocation semantics, version immutability (ADR-023, ADR-024)
4. **Cryptographic Proofs**: Ed25519 signatures, Merkle trees (Phase 4.6 hardening)

The tests should be redesigned to test what they claim: **semantic search quality** for the governance use case, not "security."

---

## Part 1: What The Tests Actually Do

### File Location
`CAPABILITY/TESTBENCH/cassette_network/adversarial/test_negative_controls.py`

### Test Classes

#### 1. TestNegativeControls
Tests that semantically unrelated queries return low similarity scores.

**Test cases (from negative_controls.json):**
- NC-001: Gibberish ("xyzzy plugh quantum banana sandwich")
- NC-002: SQL injection ("SELECT * FROM users...")
- NC-006: Destructive command ("rm -rf /") - KNOWN FAILURE
- NC-009: XSS attack ("<script>alert('XSS')</script>") - KNOWN FAILURE

#### 2. TestSemanticConfusers
Tests disambiguation between same vocabulary in different contexts.

**Test cases (from semantic_confusers.json):**
- SC-001: "apartment lease contract" vs "canon contract rules C1-C13"
- SC-004: "restore iPhone from backup" vs "catalytic restore guarantee"
- SC-009: "compress images for website" vs "compressed genesis prompt"

All marked `known_edge_case` because no real agent would query these.

#### 3. TestSecurityVectors (The xfailed class)
Three hardcoded tests with strict thresholds:

```python
@pytest.mark.xfail(reason="Vocabulary overlap: 'memories' matches memory-related content")
def test_sql_injection_rejected(self, geometric_network):
    # Threshold: 0.35

@pytest.mark.xfail(reason="Vocabulary overlap: 'script'/'alert' match code content")
def test_xss_rejected(self, geometric_network):
    # Threshold: 0.30

@pytest.mark.xfail(reason="Vocabulary overlap: file paths match filesystem docs")
def test_path_traversal_rejected(self, geometric_network):
    # Threshold: 0.25
```

---

## Part 2: Why These Tests Fail (And Why That's Expected)

### The Vocabulary Overlap Problem

The tests use all-MiniLM-L6-v2, a general-purpose sentence embedding model. This model encodes text into 384-dimensional vectors based on **semantic similarity**, which includes vocabulary overlap.

**Example: SQL Injection Query**
```
Query: "'; DELETE FROM memories WHERE '1'='1"
```

The word "memories" in this SQL injection matches documentation about the memory system:
- `MEMORY/` bucket documentation
- `memory_save()`, `memory_query()` API docs
- Session memory management

**Result:** 0.53 similarity (above 0.35 threshold)

**This is not a security vulnerability.** It's the embedding model correctly identifying that both texts contain "memories."

### Why This Is NOT A Security Issue

1. **Agents don't query with attack payloads.** Real governance queries are:
   - "What is ADR-039?"
   - "Show me INV-001 through INV-008"
   - "What are the canon contract rules?"

2. **Query results don't grant capabilities.** Even if an agent gets a result, the result is just text. The agent cannot:
   - Execute SQL on the governance system
   - Run XSS scripts
   - Perform path traversal

3. **The security boundary is elsewhere.** See Part 4.

---

## Part 3: The Real Security Model

### Trust Boundary Architecture (ADR-026)

From `LAW/CONTEXT/decisions/ADR-026-router-model-trust-boundary.md`:

```
Principle: Models are Replaceable and Untrusted

No router or model output is trusted by default. The governance system validates everything:

1. Receipt Requirements: Every router execution MUST generate receipts
2. Fail-Closed Validation: Router outputs MUST be rejected if:
   - Router produces stderr
   - Router output exceeds size cap
   - Plan fails schema validation
   - Plan attempts capability escalation
   - Router exits with non-zero code
3. Replaceability: Swapping models/routers does NOT change security model
4. No Authority Granted: Routers/models cannot bypass capability checks
```

### Security Layers

| Layer | What It Protects | How |
|-------|------------------|-----|
| Capability System | Privilege escalation | Revocation semantics, version pinning |
| Schema Validation | Malformed inputs | JSON schema enforcement, fail-closed |
| Cryptographic Proofs | Data integrity | Ed25519 signatures, Merkle trees |
| Write Firewall | Unauthorized writes | Path validation, GuardedWriter |
| Trust Boundary | Untrusted executables | Receipt verification, stderr rejection |

### Actual Security Tests That Work

From `CAPABILITY/TESTBENCH/integration/test_phase_4_6_security_hardening.py`:

```python
class TestKeyZeroization:
    """Tests for key zeroization (4.6.1)."""
    # Tests that private keys are cleared from memory after use

class TestConstantTimeComparison:
    """Tests for constant-time comparison (4.6.2)."""
    # Tests that hash comparisons don't leak timing information

class TestTOCTOUMitigation:
    """Tests for TOCTOU mitigation (4.6.3)."""
    # Tests that file operations minimize race condition windows

class TestErrorSanitization:
    """Tests for error sanitization (4.6.4)."""
    # Tests that error messages don't expose internal details
```

These test **actual security properties** at the cryptographic layer.

---

## Part 4: What These Tests SHOULD Be Testing

### Option A: Rename and Reframe as Embedding Quality Tests

The tests are valid as **embedding model quality benchmarks**, not security tests.

**Proposed Changes:**

1. Rename class from `TestSecurityVectors` to `TestEmbeddingQualityBenchmarks`
2. Change docstring from "Security-focused negative control tests" to "Embedding model vocabulary overlap benchmarks"
3. Remove xfail markers and replace with informative assertions:

```python
class TestEmbeddingQualityBenchmarks:
    """
    Benchmark tests for embedding model vocabulary discrimination.

    These tests document the expected behavior of all-MiniLM-L6-v2 when
    queries contain vocabulary that overlaps with the governance corpus.

    NOT SECURITY TESTS: These do not test security boundaries.
    The security model is enforced via fail-closed validation, not
    semantic search filtering.
    """

    def test_sql_injection_vocabulary_overlap(self, geometric_network):
        """Document vocabulary overlap behavior for SQL-like queries."""
        # Test that 'memories' in SQL matches memory docs
        # This is EXPECTED behavior, not a failure
```

### Option B: Create Actual Security Tests

If security testing is desired, test the **actual security boundaries**:

```python
class TestCassetteNetworkSecurityBoundaries:
    """Security tests for cassette network trust boundaries."""

    def test_query_results_are_readonly(self, geometric_network):
        """Query results cannot modify cassette state."""
        results = geometric_network.query("malicious query")
        # Verify no state change occurred

    def test_malformed_input_rejected(self, geometric_network):
        """Malformed queries are rejected gracefully."""
        # Test null bytes, extremely long strings, invalid encoding

    def test_query_cannot_access_external_paths(self, geometric_network):
        """Queries cannot access paths outside cassette scope."""
        # Verify path traversal in queries doesn't affect file access
```

### Option C: Remove TestSecurityVectors Entirely

The tests provide no value because:
1. They don't test security (see Part 3)
2. They document expected embedding model behavior
3. The behavior is already documented in negative_controls.json

The `known_failure` and `known_edge_case` flags in the JSON fixtures already document this behavior without requiring xfailed tests.

---

## Part 5: Proper Security Test Design Principles

### What Security Tests Should Verify

1. **Boundary Enforcement**
   - Inputs crossing trust boundaries are validated
   - Invalid inputs are rejected (fail-closed)
   - Error messages don't leak sensitive information

2. **Capability Constraints**
   - Revoked capabilities cannot be used
   - Version pinning is enforced
   - Escalation attempts are blocked

3. **Cryptographic Integrity**
   - Signatures are verified before trust
   - Hashes match expected values
   - Keys are properly managed

4. **Audit Trail**
   - All operations are receipted
   - Receipts are tamper-evident
   - Chain integrity is verifiable

### What Security Tests Should NOT Test

1. **Embedding model vocabulary discrimination** - This is ML quality, not security
2. **Whether attack payloads "match" content** - Matching is not executing
3. **Semantic similarity of off-topic queries** - No agent queries these

---

## Part 6: Related Documentation

### ADRs Related to Security
- ADR-026: Router/Model Trust Boundary
- ADR-023: Capability Revocation Semantics
- ADR-024: Capability Versioning Immutability
- ADR-020: Admission Control Gate
- ADR-019: Preflight Freshness Gate

### Security Implementation
- `LAW/CANON/POLICY/SECURITY.md` - Security policy and trust boundaries
- `CAPABILITY/PRIMITIVES/timing_safe.py` - Constant-time comparisons
- `CAPABILITY/PRIMITIVES/secure_memory.py` - Key zeroization
- `CAPABILITY/TESTBENCH/integration/test_phase_4_6_security_hardening.py` - Real security tests

### Cassette Network Theory
- `LAW/CANON/SEMANTIC/CASSETTE_NETWORK_THEORY.md` - Compression stack, not security
- `NAVIGATION/CORTEX/network/CHANGELOG.md` - Documents the test suite creation

---

## Part 7: Recommended Actions

### Immediate (No Code Changes Required)
1. Accept xfail as correct behavior documentation
2. Update test docstrings to clarify these are NOT security tests
3. Add comments explaining vocabulary overlap is expected

### Short-Term (Minor Refactoring)
1. Rename `TestSecurityVectors` to `TestEmbeddingVocabularyBenchmarks`
2. Remove misleading "security" terminology from test names
3. Add cross-reference to actual security tests in Phase 4.6

### Medium-Term (If Security Testing Desired)
1. Create `test_cassette_security_boundaries.py` with real security tests
2. Test fail-closed validation, input sanitization, path constraints
3. Integrate with existing Phase 4.6 security hardening tests

---

## Conclusion

The xfailed tests in `test_negative_controls.py` are not broken - they are correctly documenting that:

1. **Embedding models match vocabulary, not intent** - This is fundamental to how sentence embeddings work
2. **The cassette network is a compression/retrieval layer, not a security boundary** - Security is enforced elsewhere
3. **No agent would query with attack payloads** - The tests document unrealistic edge cases

The real fix is not to change the tests or the embedding model, but to:
1. **Clarify the documentation** to explain what these tests actually measure
2. **Rename the test class** to remove misleading "security" terminology
3. **Reference the actual security model** in ADR-026 and Phase 4.6 hardening

**The cassette network is secure because untrusted inputs are validated, not because semantic search rejects attack-like strings.**

---

*Investigation completed by Claude Opus 4.5 on 2026-01-25*
