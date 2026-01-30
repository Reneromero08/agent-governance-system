# Q51 Master Investigation Plan

**Objective**: Find definitive answers to Q51 by fixing methodological flaws and creating rigorous new tests

**Location**: THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/COMPROMISED/

**Status**: Build mode - can create files and run tests

## Critical Flaws to Fix

### FORMULA Q51 Flaws:
1. **Zero Signature**: Tests uniform distribution, not 8th roots
2. **Pinwheel**: V=0.27 below threshold, mislabeled as "PARTIAL"
3. **Phase Arithmetic**: 90.9% is good but doesn't prove complex structure
4. **Berry Holonomy**: Q=1.0 suspicious, measures winding not Berry phase

### kimi Q51 Flaws:
1. **8e Universality**: Vocabulary artifact (36% error due to word choice)
2. **Berry Phase**: Correctly notes it's undefined, but doesn't explain phase-like patterns
3. **Missing**: No explanation for why FORMULA found phase arithmetic works

## New Rigorous Tests to Create

### Test 1: Definitive Phase Structure
- **Purpose**: Distinguish geometric angle vs complex phase
- **Method**: 
  - Compute PCA angles (geometric)
  - Check for complex conjugate pairs in spectrum
  - Test if angles are coordinate-dependent (geometric) or invariant (physical)
- **Success**: Determines if "phase" is real geometry or complex structure

### Test 2: Vocabulary-Robust 8e
- **Purpose**: Fix vocabulary sensitivity
- **Method**:
  - Use 500+ word standardized vocabulary
  - Test multiple random samples
  - Report CV and convergence
- **Success**: Stable 8e estimate across samples

### Test 3: Semantic Loop Topology
- **Purpose**: Distinguish winding number vs Berry phase
- **Method**:
  - Create semantic loops (king→queen→woman→man→king)
  - Measure in 2D, 3D, and full dimension
  - Check if winding is coordinate-dependent
- **Success**: Determines if topological or geometric

### Test 4: Phase Addition Validation
- **Purpose**: Validate if phase arithmetic proves complex structure
- **Method**:
  - Test analogies with known multiplicative structure
  - Test non-analogies
  - Use negative controls (random words)
  - Check if success rate distinguishes real vs complex
- **Success**: Determines if phase arithmetic is unique to complex

### Test 5: Coordinate Independence
- **Purpose**: Test if findings are coordinate-dependent or invariant
- **Method**:
  - Rotate PCA basis randomly
  - Recompute all metrics
  - Check which are invariant (physical) vs dependent (geometric)
- **Success**: Separates real geometric artifacts from physical structure

## Execution Plan

### Phase 1: Fix Existing Tests (Day 1)
- [ ] Fix 8e universality with larger vocabulary
- [ ] Re-run with proper statistics
- [ ] Document vocabulary effects

### Phase 2: Create New Tests (Days 2-3)
- [ ] Test 1: Definitive Phase Structure
- [ ] Test 2: Vocabulary-Robust 8e
- [ ] Test 3: Semantic Loop Topology
- [ ] Test 4: Phase Addition Validation
- [ ] Test 5: Coordinate Independence

### Phase 3: Large-Scale Validation (Days 4-5)
- [ ] Run all tests on 5+ models
- [ ] Collect statistics
- [ ] Cross-model comparison

### Phase 4: Synthesis (Day 6)
- [ ] Meta-analysis
- [ ] Final verdict
- [ ] Documentation

## Subagent Tasks

1. **Subagent A**: Fix 8e universality + Test 2
2. **Subagent B**: Test 1 (Definitive Phase Structure)
3. **Subagent C**: Test 3 + Test 5 (Topology + Coordinate Independence)
4. **Subagent D**: Test 4 (Phase Addition Validation)
5. **Subagent E**: Meta-analysis and synthesis

All subagents work in COMPROMISED only.
