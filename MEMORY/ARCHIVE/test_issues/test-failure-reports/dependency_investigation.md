# Dependency Investigation Report

**Date**: 2026-01-25
**Status**: COMPLETE
**Scope**: Full repository dependency audit

---

## Executive Summary

The repository has a **fragmented dependency structure** with multiple `requirements.txt` files that are NOT consolidated. The root `requirements.txt` contains only minimal dependencies while module-specific requirements files contain the actual dependencies needed. **CI installs only the root requirements**, causing tests that depend on CORTEX/LAB dependencies to be skipped via graceful degradation patterns.

**Key Finding**: Dependencies ARE installed in the active Python environment but tests STILL skip because:
1. CI workflow only installs root `requirements.txt`
2. Module-specific `requirements.txt` files are not installed in CI
3. Tests use graceful degradation (`try/except ImportError`) which silently skips

---

## Dependency File Structure

### 1. Root requirements.txt
**Location**: `D:\CCC 2.0\AI\agent-governance-system\requirements.txt`
**Content**:
```
jsonschema>=4.18.0
pyyaml>=6.0
numpy>=1.24.0
```

**Issue**: This is MINIMAL. It lacks:
- `sentence-transformers` (needed by CORTEX, CAT_CHAT, FORMULA experiments)
- `torch` (needed by CORTEX)
- `scipy` (needed by experiments)
- `scikit-learn` (needed by experiments)
- `tiktoken` (needed by semiotic compression tests)
- `networkx` (needed by experiments)

### 2. NAVIGATION/CORTEX/requirements.txt
**Location**: `D:\CCC 2.0\AI\agent-governance-system\NAVIGATION\CORTEX\requirements.txt`
**Content**:
```
# CORTEX Semantic Core Requirements
# Related: ADR-030 (Semantic Core + Translation Layer Architecture)

# Vector embeddings
sentence-transformers>=2.2.0
torch>=2.0.0
numpy>=1.24.0

# Optional: FAISS for large-scale vector search (uncomment if needed)
# faiss-cpu>=1.7.0

# Database
# SQLite is built into Python, but you may want pysqlite3 for newer features
# pysqlite3>=0.5.0
```

**Status**: NOT installed by CI. Contains the critical `sentence-transformers` and `torch` deps.

### 3. THOUGHT/LAB/FORMULA/experiments/requirements.txt
**Location**: `D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\experiments\requirements.txt`
**Content**:
```
# Formula Falsification Test Suite Dependencies
# Install: pip install -r requirements.txt

# Core
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Embeddings (for semantic tests)
sentence-transformers>=2.0.0

# Network analysis
networkx>=2.6.0

# Audio (optional)
librosa>=0.9.0

# Visualization (optional)
matplotlib>=3.4.0
seaborn>=0.11.0

# Token counting (optional)
tiktoken>=0.3.0
```

**Status**: NOT installed by CI. Contains research/experiment dependencies.

### 4. THOUGHT/LAB/FERAL_RESIDENT/dashboard/requirements.txt
**Location**: `D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FERAL_RESIDENT\dashboard\requirements.txt`
**Content**:
```
# Feral Dashboard Dependencies

# Web server
fastapi>=0.104.0
uvicorn>=0.24.0
websockets>=12.0

# Already required by Feral Resident
# numpy
# sentence-transformers
# torch
```

**Status**: NOT installed by CI. Dashboard-specific deps.

### 5. MCP Builder requirements.txt (Deprecated)
**Location**: `D:\CCC 2.0\AI\agent-governance-system\MEMORY\ARCHIVE\skills-deprecated\mcp-builder\scripts\requirements.txt`
**Content**:
```
anthropic>=0.39.0
mcp>=1.1.0
```

**Status**: Archived/deprecated.

---

## CI Workflow Analysis

### .github/workflows/contracts.yml

The CI workflow installs dependencies via:
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install pytest cryptography referencing
```

**Issues**:
1. Only installs ROOT `requirements.txt` (3 packages)
2. Does NOT install `NAVIGATION/CORTEX/requirements.txt`
3. Does NOT install `THOUGHT/LAB/FORMULA/experiments/requirements.txt`
4. Manually adds `pytest cryptography referencing` but misses `sentence-transformers`, `torch`, `tiktoken`

### Tests Explicitly Ignored in CI
The CI workflow explicitly ignores tests that require semantic embeddings:
```yaml
--ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_1_canon_embedding.py
--ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_2_adr_embedding.py
--ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_3_model_registry.py
--ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_1_4_skill_discovery.py
--ignore=CAPABILITY/TESTBENCH/integration/test_phase_5_2_3_stacked_resolution.py
```

This is a WORKAROUND for missing dependencies, not a real solution.

---

## Graceful Degradation Patterns

Tests use try/except patterns to skip when dependencies are missing:

### Pattern 1: REAL_EMBEDDINGS flag
```python
try:
    from sentence_transformers import SentenceTransformer
    REAL_EMBEDDINGS = True
except ImportError:
    REAL_EMBEDDINGS = False

@pytest.mark.skipif(not REAL_EMBEDDINGS, reason="Requires sentence-transformers")
def test_semantic_rehydration(self):
    ...
```

**Files using this pattern**:
- `THOUGHT/LAB/CAT_CHAT/tests/test_catalytic_stress.py`
- `THOUGHT/LAB/FORMULA/experiments/open_questions/q10/test_q10_alignment_detection.py`
- `THOUGHT/LAB/FORMULA/experiments/open_questions/q17/test_q17_r_gate.py`

### Pattern 2: TIKTOKEN_AVAILABLE flag
```python
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

@pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
def test_symbolic_vs_expanded_compression(self):
    ...
```

**Files using this pattern**:
- `CAPABILITY/TESTBENCH/integration/test_phase_5_2_semiotic_compression.py`
- `CAPABILITY/TOOLS/scl/run_scl_proof.py`
- `NAVIGATION/CORTEX/network/spc_metrics.py`

### Pattern 3: pytest.importorskip
```python
@pytest.fixture
def tiktoken_encoder():
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.get_encoding("cl100k_base")
```

**Files using this pattern**:
- `CAPABILITY/TESTBENCH/integration/test_phase_5_2_1_macro_grammar.py`

---

## Current Local Environment Status

**Active Python**: `C:\Users\rene_\AppData\Local\Programs\Python\Python311\python.exe`

**Package Status** (in global Python, NOT venv):
| Package | Status | Version |
|---------|--------|---------|
| sentence-transformers | INSTALLED | 3.3.1 |
| torch | INSTALLED | 2.9.1+cpu |
| numpy | INSTALLED | 2.0.2 |
| scipy | INSTALLED | 1.17.0 |
| scikit-learn | INSTALLED | 1.8.0 |
| tiktoken | INSTALLED | 0.12.0 |
| networkx | INSTALLED | (available) |
| jsonschema | INSTALLED | (available) |
| PyYAML | INSTALLED | 6.0.1 |
| cryptography | INSTALLED | (available) |
| matplotlib | INSTALLED | (available) |
| librosa | MISSING | - |
| seaborn | MISSING | - |

**Key Insight**: Most dependencies ARE installed locally but CI doesn't have them.

---

## The Real Problem

### Why Tests Skip When Dependencies Should Be Present

1. **Local environment**: Dependencies ARE installed in the global Python
2. **CI environment**: Only root `requirements.txt` installed (3 packages)
3. **Graceful degradation**: Tests skip silently instead of failing hard

### Architectural Debt

The repository has evolved with multiple subsystems (CORTEX, CAT_CHAT, FORMULA experiments) each with their own dependencies, but:
- No consolidated `requirements-full.txt` or `requirements-dev.txt`
- No `setup.py` or `pyproject.toml` at root level declaring dependencies
- CI workflow uses explicit `--ignore` flags as a band-aid

---

## The Real Fix

### Option A: Consolidated requirements-dev.txt (RECOMMENDED)

Create `requirements-dev.txt` at repo root:
```
# Core (from root requirements.txt)
jsonschema>=4.18.0
pyyaml>=6.0
numpy>=1.24.0

# CORTEX (from NAVIGATION/CORTEX/requirements.txt)
sentence-transformers>=2.2.0
torch>=2.0.0

# Testing
pytest>=7.0.0
cryptography>=41.0.0
referencing>=0.30.0

# Token counting
tiktoken>=0.5.0

# Research/experiments (optional but recommended)
scipy>=1.7.0
scikit-learn>=1.0.0
networkx>=2.6.0
matplotlib>=3.4.0
```

Update CI to install: `pip install -r requirements-dev.txt`

### Option B: pyproject.toml with optional dependencies

Create `pyproject.toml`:
```toml
[project]
name = "agent-governance-system"
version = "1.0.0"
requires-python = ">=3.10"

dependencies = [
    "jsonschema>=4.18.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
cortex = [
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
]
dev = [
    "pytest>=7.0.0",
    "cryptography>=41.0.0",
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "tiktoken>=0.5.0",
]
experiments = [
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "networkx>=2.6.0",
]
```

Install in CI: `pip install -e .[dev]`

### Option C: Update CI workflow only

Modify `.github/workflows/contracts.yml`:
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -r NAVIGATION/CORTEX/requirements.txt
    python -m pip install pytest cryptography referencing tiktoken
```

---

## Recommendation

**Implement Option A (Consolidated requirements-dev.txt)** because:
1. Minimal change to existing structure
2. Single file to maintain for CI
3. Does not require pyproject.toml migration
4. Can be implemented in one commit

Then remove the `--ignore` flags from CI once dependencies are properly installed.

---

## Files That Need Updates

1. **NEW**: `requirements-dev.txt` (consolidated dev dependencies)
2. **UPDATE**: `.github/workflows/contracts.yml` (use requirements-dev.txt)
3. **OPTIONAL**: Remove explicit `--ignore` flags from CI once deps are installed

---

## Verification Steps

After fix:
```bash
# Install dev deps
pip install -r requirements-dev.txt

# Run full test suite without ignores
pytest CAPABILITY/TESTBENCH/ -v

# Verify no skips due to missing deps
pytest CAPABILITY/TESTBENCH/ -v -rs 2>&1 | grep -i "skipped"
```

Expected: Zero skips due to "sentence-transformers not installed" or "tiktoken not installed"

---

## Appendix: All requirements.txt Files Found

| Location | Purpose | CI Installed |
|----------|---------|--------------|
| `./requirements.txt` | Root minimal deps | YES |
| `NAVIGATION/CORTEX/requirements.txt` | CORTEX semantic core | NO |
| `THOUGHT/LAB/FORMULA/experiments/requirements.txt` | Research experiments | NO |
| `THOUGHT/LAB/FERAL_RESIDENT/dashboard/requirements.txt` | Dashboard server | NO |
| `MEMORY/ARCHIVE/skills-deprecated/mcp-builder/scripts/requirements.txt` | Deprecated MCP builder | NO |
| Multiple fixture copies under `MEMORY/LLM_PACKER/_packs/` | Test fixtures (copies) | N/A |

---

**Report Generated By**: Claude Opus 4.5
**Investigation Date**: 2026-01-25
