# Task 3.1: Router & Fallback Stability - Completion Report

## Status: ✅ PASS

## Summary

Implemented **3.1.1 Stabilize model router: deterministic selection + explicit fallback chain (Z.3.1)** by creating a standalone model router module with comprehensive test coverage.

## Checklist Coverage

### 3.1.1 Stabilize model router: deterministic selection + explicit fallback chain (Z.3.1)

**Status**: ✅ COMPLETE

**Evidence**:
1. Created `CAPABILITY/TOOLS/model_router.py` - deterministic model selection module
2. Created `CAPABILITY/TESTBENCH/core/test_model_router.py` - 32 comprehensive tests (all passing)
3. Implemented explicit fallback chain support with index-based selection
4. Deterministic chain hashing for reproducibility

## What Changed

### Files Created

1. **[CAPABILITY/TOOLS/model_router.py](CAPABILITY/TOOLS/model_router.py)** (8,963 bytes)
   - Core router implementation
   - `KNOWN_MODELS` registry with 6 models (Claude, GPT, Gemini)
   - `select_model()` function for deterministic selection
   - `validate_model()` for fail-closed model validation
   - `create_router_receipt()` for auditing
   - Chain hash computation for reproducibility

2. **[CAPABILITY/TESTBENCH/core/test_model_router.py](CAPABILITY/TESTBENCH/core/test_model_router.py)** (11,882 bytes)
   - 32 test cases across 7 test classes
   - Tests for parsing, validation, selection, determinism
   - Integration tests for full workflow
   - Registry validation tests

### Files Modified

- None (clean implementation, no existing code modified)

## Implementation Details

### Model Router Architecture

**Key Components**:

1. **ModelSpec** (dataclass, frozen):
   - `name`: Human-readable model name
   - `model_id`: Canonical identifier for API calls
   - `reasoning_level`: "Low", "Medium", "High", or "Deep"
   - `thinking_mode`: Boolean flag for reasoning output

2. **KNOWN_MODELS Registry**:
   - Single source of truth for valid models
   - Claude Sonnet 4.5 (non-thinking) - Medium reasoning
   - Claude Sonnet - High reasoning
   - Claude Opus 4.5 - Deep reasoning with thinking
   - GPT-5.2-Codex - High reasoning
   - Gemini Pro - High reasoning
   - Gemini 3 Pro (Low) - Low reasoning

3. **select_model() Function**:
   ```python
   select_model(
       primary_model: str,
       fallback_chain: Optional[List[str]] = None,
       selection_index: int = 0
   ) -> RouterSelection
   ```
   - Deterministic: same inputs always produce same output
   - Explicit fallback: chain[0] = primary, chain[1+] = fallbacks
   - Fail-closed: invalid models raise `InvalidModelError`

4. **RouterSelection Result**:
   - `selected_model`: The chosen ModelSpec
   - `selection_index`: Position in chain (0 = primary, 1+ = fallback)
   - `selection_reason`: "primary_model" or "fallback[N]"
   - `fallback_chain_hash`: SHA256 of canonical chain JSON

5. **Error Handling**:
   - `RouterError`: Base exception
   - `InvalidModelError`: Unknown model name
   - `EmptyFallbackChainError`: No models specified

### Determinism Guarantees

1. **Chain Hash Determinism**:
   - SHA256 of canonical JSON: `{"primary": "...", "fallbacks": [...]}`
   - Same chain → same hash (verified in tests)
   - Different chains → different hashes
   - Order matters (verified in tests)

2. **Selection Determinism**:
   - Given (primary, fallback_chain, selection_index) → always same result
   - Verified across 10 runs in `test_selection_determinism_across_runs`

3. **Model Name Parsing**:
   - Strips reasoning annotations: "Claude (Reasoning: High)" → "Claude"
   - Idempotent parsing (same input → same output)

## Proofs (Tests + Commands)

### Test Results

```
python -m pytest CAPABILITY/TESTBENCH/core/test_model_router.py -v
============================= test session starts =============================
platform win32 -- Python 3.11.6, pytest-9.0.2, pluggy-1.6.0
collected 32 items

CAPABILITY/TESTBENCH/core/test_model_router.py::TestParseModelName::test_parse_simple_name PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestParseModelName::test_parse_with_reasoning_annotation PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestParseModelName::test_parse_empty_string PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestParseModelName::test_parse_only_reasoning PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestValidateModel::test_validate_known_model PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestValidateModel::test_validate_with_reasoning_annotation PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestValidateModel::test_validate_unknown_model PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestValidateModel::test_validate_empty_model PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestValidateModel::test_all_known_models_valid PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_select_primary_only PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_select_first_fallback PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_select_second_fallback PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_select_out_of_bounds PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_select_negative_index PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_empty_chain_error PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_invalid_primary_model PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_invalid_fallback_model PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestSelectModel::test_fallback_chain_none_is_empty_list PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestDeterminism::test_chain_hash_determinism PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestDeterminism::test_different_chain_different_hash PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestDeterminism::test_order_matters_for_hash PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestDeterminism::test_selection_determinism_across_runs PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestCreateRouterReceipt::test_create_receipt_basic PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestCreateRouterReceipt::test_receipt_with_no_fallbacks PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestModelSpecs::test_model_spec_frozen PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestModelSpecs::test_model_spec_to_dict PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestKnownModelsRegistry::test_registry_has_claude_models PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestKnownModelsRegistry::test_registry_has_gpt_models PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestKnownModelsRegistry::test_registry_has_gemini_models PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestKnownModelsRegistry::test_all_specs_are_frozen PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestIntegration::test_full_workflow_primary_success PASSED
CAPABILITY/TESTBENCH/core/test_model_router.py::TestIntegration::test_full_workflow_fallback_selection PASSED

============================= 32 passed in 0.06s ==============================
```

**Result**: ✅ All 32 tests passed

### Regression Testing

```
python -m pytest CAPABILITY/TESTBENCH/core/ -q --tb=short
..................................................................
66 passed in 0.90s
```

**Result**: ✅ No regressions (66 tests passed including 32 new router tests)

## Deviations

**None**. Implementation follows the task specification exactly:
- ✅ Deterministic model selection
- ✅ Explicit fallback chain support
- ✅ Fail-closed validation
- ✅ Comprehensive test coverage
- ✅ Receipt generation for auditing

## Scope Compliance

### Allowed Writes (all within scope):
- `CAPABILITY/TOOLS/model_router.py` - core implementation
- `CAPABILITY/TESTBENCH/core/test_model_router.py` - tests
- `LAW/CONTRACTS/_runs/_tmp/prompts/3.1_router-fallback-stability/receipt.json` - receipt
- `LAW/CONTRACTS/_runs/_tmp/prompts/3.1_router-fallback-stability/REPORT.md` - this report
- `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md` - mark task complete (pending)

### No Forbidden Writes:
- ✅ No changes to LAW/CANON/
- ✅ No changes to .git/
- ✅ No changes to BUILD/
- ✅ No cleanup sweeps outside task scope

## Integration Points

### Current State

The model router is **ready to use** but not yet integrated into execution flows. Future integration points:

1. **Prompt Runner** (`CAPABILITY/SKILLS/utilities/prompt-runner/run.py`):
   - Already parses `primary_model` and `fallback_chain` from YAML frontmatter
   - Can import and call `select_model()` to route execution
   - Can emit `RouterSelection` in receipts

2. **CAT Chat** (`THOUGHT/LAB/CAT_CHAT/catalytic_chat/`):
   - Planner can use router to select models for task execution
   - Executor can validate model selections against registry

3. **AGS Tool** (`CAPABILITY/TOOLS/ags.py`):
   - Can use router for plan generation with model-specific routers

### CLI Usage

```bash
# Test router directly
python CAPABILITY/TOOLS/model_router.py "Claude Sonnet 4.5 (non-thinking)" "Claude Sonnet" "Gemini Pro"

# Output (JSON):
{
  "selected_model": {
    "name": "Claude Sonnet 4.5 (non-thinking)",
    "model_id": "claude-sonnet-4-5",
    "reasoning_level": "Medium",
    "thinking_mode": false
  },
  "selection_index": 0,
  "selection_reason": "primary_model",
  "fallback_chain_hash": "8f4e3c2..."
}
```

## Next Recommended Section

**Task 3.2: Memory Integration (Z.3.2)**

Now that model routing is stabilized, the next logical step is implementing CAT Chat context window management to enable the router to work with real chat sessions.

## Reproduction Steps

To verify this implementation:

```bash
# 1. Run router tests
python -m pytest CAPABILITY/TESTBENCH/core/test_model_router.py -v

# 2. Run full core test suite (regression check)
python -m pytest CAPABILITY/TESTBENCH/core/ -q

# 3. Test CLI directly
python CAPABILITY/TOOLS/model_router.py "Claude Sonnet 4.5 (non-thinking)" "Claude Sonnet"

# 4. Verify determinism (run twice, compare hashes)
python CAPABILITY/TOOLS/model_router.py "Claude Sonnet 4.5 (non-thinking)" "Gemini Pro" > /tmp/run1.json
python CAPABILITY/TOOLS/model_router.py "Claude Sonnet 4.5 (non-thinking)" "Gemini Pro" > /tmp/run2.json
diff /tmp/run1.json /tmp/run2.json  # Should be identical
```

## Artifacts

- Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/3.1_router-fallback-stability/receipt.json`
- Report: `LAW/CONTRACTS/_runs/_tmp/prompts/3.1_router-fallback-stability/REPORT.md` (this file)
- Implementation: `CAPABILITY/TOOLS/model_router.py`
- Tests: `CAPABILITY/TESTBENCH/core/test_model_router.py`

---

**Task 3.1: COMPLETE** ✅

All checklist items satisfied. All tests passing. No scope drift. Ready for next section.
