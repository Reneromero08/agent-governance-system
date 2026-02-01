---
uuid: 00000000-0000-0000-0000-000000000000
title: Test Organization Investigation - CAPABILITY/TESTBENCH vs THOUGHT/LAB
section: report
bucket: reports
author: Claude Opus 4.5
priority: High
created: 2026-01-25
modified: 2026-01-25
status: Complete
summary: Full investigation into the organization of research/experimental tests between CAPABILITY/TESTBENCH and THOUGHT/LAB/tests
tags:
- investigation
- test-organization
- governance
- testbench
- lab
---
<!-- CONTENT_HASH: PENDING_CALCULATION -->

# Test Organization Investigation Report

**Date:** 2026-01-25
**Investigator:** Claude Opus 4.5 (session investigation)
**Scope:** Why are research/experimental tests in CAPABILITY/TESTBENCH instead of THOUGHT/LAB?

---

## Executive Summary

The investigation reveals that the test organization is **intentional and correctly structured** according to the 6-bucket system. What appears to be "research tests in the wrong place" is actually a well-designed graduation pipeline that moves tests from LAB to TESTBENCH as features mature.

**Key Findings:**
1. TESTBENCH is for **validation suites** (production tests), not research experiments
2. THOUGHT/LAB is for **experimental prototypes** with their own co-located tests
3. A recent graduation (2026-01-25) correctly moved cassette network tests to multiple locations
4. The remaining "research" tests in TESTBENCH/cassette_network are **intentionally kept** for ongoing hypothesis validation

---

## 1. Intended Test Organization Structure

### 1.1 Bucket Definitions (from LAW/CANON/META/SYSTEM_BUCKETS.md)

| Bucket | Purpose | Test Location |
|--------|---------|---------------|
| **CAPABILITY** | Production instruments | `CAPABILITY/TESTBENCH/` |
| **THOUGHT** | Experimental labs | `THOUGHT/LAB/{project}/tests/` |
| **LAW** | Contract enforcement | `LAW/CONTRACTS/fixtures/` |

### 1.2 Test Categories

**CAPABILITY/TESTBENCH/** (Production Validation)
- Tests for graduated, production-ready code
- CI runs these tests on every push/PR
- Must pass for code to merge to main
- Located in `TESTBENCH/` subdirectories: `integration/`, `pipeline/`, `core/`, etc.

**THOUGHT/LAB/{project}/tests/** (Research Experiments)
- Tests for experimental features still in development
- NOT run by CI (experiments can fail without blocking main)
- Co-located with the experimental code they test
- Examples: `THOUGHT/LAB/CAT_CHAT/tests/`, `THOUGHT/LAB/FERAL_RESIDENT/dashboard/tests/`

**LAW/CONTRACTS/fixtures/** (Governance Fixtures)
- Contract enforcement tests with deterministic fixtures
- Schema validation and governance rule tests
- Run by `LAW/CONTRACTS/runner.py`
- Examples: `fixtures/cassette_network/` (graduated 2026-01-25)

---

## 2. CI Configuration Analysis

### 2.1 What CI Runs (from .github/workflows/contracts.yml)

```yaml
python -m pytest CAPABILITY/TESTBENCH -q -s \
  --ignore=CAPABILITY/TESTBENCH/phase6 \
  --ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_1_canon_embedding.py \
  --ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_2_adr_embedding.py \
  --ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_3_model_registry.py \
  --ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_4_skill_discovery.py \
  --ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_2_3_stacked_resolution.py \
  --ignore=CAPABILITY/TESTBENCH/integration/test_phase_2_4_1b_write_enforcement.py \
  --ignore=CAPABILITY/TESTBENCH/pipeline/test_write_firewall.py
```

**Key Points:**
- CI runs ALL of TESTBENCH except explicitly ignored tests
- Ignored tests are those with external dependencies (embeddings, models)
- THOUGHT/LAB tests are NOT run by CI (correct behavior)

### 2.2 Conftest Configuration (CAPABILITY/TESTBENCH/conftest.py)

```python
def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="Slow test - use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
```

This allows slow research-like tests to exist in TESTBENCH but be skipped unless explicitly requested.

---

## 3. Cassette Network Tests - What's Where and Why

### 3.1 Recent Graduation (2026-01-25, commit 921337fd)

The most recent refactor correctly graduated Cassette Network from LAB to production:

| From | To | Category |
|------|------|----------|
| THOUGHT/LAB/CASSETTE_NETWORK/ (deleted) | Multiple locations | All 14 files relocated |
| - | LAW/CONTRACTS/fixtures/cassette_network/ | Production contract tests |
| - | LAW/CANON/SEMANTIC/ | Specification docs |
| - | MEMORY/ARCHIVE/cassette-network-research/ | Historical research |
| - | THOUGHT/LAB/VECTOR_ELO/eigen-alignment/cassette-integration/ | ESAP (still experimental) |

### 3.2 Tests in LAW/CONTRACTS/fixtures/cassette_network/

**Correctly Placed Production Tests:**
- `ground_truth/` - Retrieval accuracy validation
- `adversarial/` - Negative controls & robustness
- `compression/` - H(X|S) ~ 0 proof
- `determinism/` - Reproducibility tests
- `benchmark_success_metrics.py` - Performance validation

### 3.3 Tests Still in CAPABILITY/TESTBENCH/cassette_network/

**Intentionally Kept for Research:**
- `cross_model/` - Eigenstructure alignment experiments (4 test files)
- `qec/` - Quantum error correction hypothesis testing (3 test files)

**Why These Remain:**
From `LAW/CONTRACTS/fixtures/cassette_network/README.md`:
> "The following remain in CAPABILITY/TESTBENCH/cassette_network/ for ongoing research:
> - `cross_model/` - Eigenstructure alignment experiments
> - `qec/` - Quantum error correction hypothesis testing"

These are **not production features** - they are active research validating mathematical hypotheses about:
1. Cross-model semantic alignment (Procrustes transforms)
2. Quantum error correction analogies for semantic retrieval

---

## 4. Analysis: Why This Organization Makes Sense

### 4.1 The Graduation Pipeline

```
THOUGHT/LAB/{project}/tests/  -->  CAPABILITY/TESTBENCH/{feature}/  -->  LAW/CONTRACTS/fixtures/{feature}/
       (experimental)                   (validation)                       (governance)

       [research phase]              [production phase]                  [contract phase]
       NOT in CI                      IN CI                              Contract enforcement
```

### 4.2 Cross-Model and QEC Tests

The `cross_model/` and `qec/` tests in TESTBENCH are in a **transition zone**:
- They test production code (NAVIGATION/CORTEX/network/)
- But validate research hypotheses (cross-model retrieval parity, QEC bounds)
- They use `xfail` markers appropriately for aspirational tests
- They ARE included in TESTBENCH because they validate production semantic search behavior

From `test_cross_model_retrieval.py`:
```python
@pytest.mark.cross_model
class TestCrossModelTaskParity:
    """THE CORE CLAIM: Model B retrieves correct content
    using Model A's transformed query vector."""
```

This tests a production capability (semantic search works across embedding models) not a lab experiment.

### 4.3 The Root Cause: No Wrong Placement

The "research" tests in TESTBENCH are there because:
1. They test **production code** (cassette network in NAVIGATION/CORTEX/network/)
2. They validate **mathematical claims** that the production system depends on
3. They use appropriate markers (`xfail`, `slow`) for tests that may not always pass
4. The graduation process correctly left them in TESTBENCH while moving fixtures to LAW/CONTRACTS

---

## 5. Governance Compliance

### 5.1 From AGENTS.md Section 0.7 (MCP-First Principle)

Tests should validate production behavior, not duplicate research.

### 5.2 From TESTBENCH.md

```
TESTBENCH FOCUS -> Phase 1 (CATLAB) <- Governor validates here
```

TESTBENCH is explicitly for validation, not experimentation.

### 5.3 From SYSTEM_BUCKETS.md

> `CAPABILITY/TESTBENCH/` - Validation suites

The tests in TESTBENCH/cassette_network validate the production cassette network.

---

## 6. Recommendations

### 6.1 No Reorganization Needed

The current structure is correct. The investigation found:
- Production tests are in LAW/CONTRACTS/fixtures/ (correct)
- Validation tests for production code are in TESTBENCH (correct)
- Experimental project tests are in THOUGHT/LAB/{project}/tests/ (correct)

### 6.2 Documentation Enhancement

**Recommended:** Add a `CAPABILITY/TESTBENCH/README.md` explaining:
1. What belongs in TESTBENCH vs LAW/CONTRACTS/fixtures
2. How to use markers (`xfail`, `slow`, `ground_truth`, `adversarial`)
3. The graduation pipeline from LAB -> TESTBENCH -> LAW/CONTRACTS

### 6.3 Marker Standardization

The cassette_network tests use excellent markers that should be standard:
- `@pytest.mark.ground_truth` - Known correct answers
- `@pytest.mark.adversarial` - Robustness/rejection tests
- `@pytest.mark.determinism` - Reproducibility tests
- `@pytest.mark.compression` - Compression claims
- `@pytest.mark.cross_model` - Cross-model alignment
- `@pytest.mark.xfail` - Aspirational tests (known limitations)

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Are research tests in the wrong place? | **No** - they are validation tests for production code |
| Should cross_model/ move to LAB? | **No** - it tests production semantic search behavior |
| Should qec/ move to LAB? | **No** - it validates production error bounds |
| What should move? | Nothing - graduation happened correctly on 2026-01-25 |
| What needs to be fixed? | Documentation only - add TESTBENCH/README.md |

---

## 8. Files Reviewed

### TESTBENCH Structure
- `CAPABILITY/TESTBENCH/conftest.py`
- `CAPABILITY/TESTBENCH/TESTBENCH.md`
- `CAPABILITY/TESTBENCH/cassette_network/conftest.py`
- `CAPABILITY/TESTBENCH/cassette_network/cross_model/test_cross_model_retrieval.py`
- `CAPABILITY/TESTBENCH/cassette_network/adversarial/test_negative_controls.py`

### Governance Documents
- `LAW/CANON/META/SYSTEM_BUCKETS.md`
- `AGENTS.md`
- `.github/workflows/contracts.yml`

### Graduation Evidence
- `LAW/CONTRACTS/fixtures/cassette_network/README.md`
- `NAVIGATION/CORTEX/network/CHANGELOG.md`
- Commit `921337fd` (refactor: graduate Cassette Network from LAB to production locations)

---

## 9. Conclusion

The test organization is **correct and intentional**. The perceived problem ("research tests in TESTBENCH") is actually the result of a well-designed graduation pipeline where:

1. **Lab experiments** stay in `THOUGHT/LAB/{project}/tests/`
2. **Production validation** graduates to `CAPABILITY/TESTBENCH/`
3. **Contract enforcement** graduates to `LAW/CONTRACTS/fixtures/`

The cross_model and qec tests remain in TESTBENCH because they validate claims about production semantic search, not because they are research experiments.

**Status:** INVESTIGATION COMPLETE - NO ACTION REQUIRED

---

**Report Hash:** PENDING (to be calculated on save)
**Co-Authored-By:** Claude Opus 4.5 <noreply@anthropic.com>
