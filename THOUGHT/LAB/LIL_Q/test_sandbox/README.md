# Quantum Entanglement Test: Big Model vs Tiny Model

## Test Goal

Demonstrate that quantum entanglement with good context vectors enables a tiny model (0.5B parameters) to solve problems it normally can't, using problems that big models (or Claude) can solve without context.

**Hypothesis**: Context containing step-by-step solution methodology, when blended via E-weighting on the quantum manifold, enables tiny models to solve problems beyond their capability.

## Test Problems

### 1. Math - Algebraic Equation
**Problem**: Solve for x: (2x + 3)² - (x - 1)² = 45

**Knowledge Base**: expanding_squares.md, difference_of_squares.md, quadratic_formula.md

**Answer**: x ≈ 1.88 or x ≈ -6.55

### 2. Code - Debugging Fibonacci
**Problem**: Fix the bug in this function that hangs:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
result = fibonacci(40)  # Hangs forever
```

**Knowledge Base**: memoization.md, recursive_optimization.md, caching_patterns.md

**Answer**: Add @lru_cache decorator or implement memoization

### 3. Logic - Knights and Knaves
**Problem**: On an island, knights always tell truth and knaves always lie. You meet two people. A says "We are both knaves." What are A and B?

**Knowledge Base**: knights_and_knaves_strategy.md, logical_deduction.md, truth_tables.md

**Answer**: A is a knave, B is a knight

### 4. Chemistry - Balancing Equations
**Problem**: Balance this equation: Fe + O₂ → Fe₂O₃

**Knowledge Base**: balancing_equations.md, oxidation_states.md, stoichiometry.md

**Answer**: 4Fe + 3O₂ → 2Fe₂O₃

## Test Structure

```
test_sandbox/
├── docs/
│   ├── math/
│   │   ├── expanding_squares.md
│   │   ├── difference_of_squares.md
│   │   └── quadratic_formula.md
│   ├── code/
│   │   ├── memoization.md
│   │   ├── recursive_optimization.md
│   │   └── caching_patterns.md
│   ├── logic/
│   │   ├── knights_and_knaves_strategy.md
│   │   ├── logical_deduction.md
│   │   └── truth_tables.md
│   └── chemistry/
│       ├── balancing_equations.md
│       ├── oxidation_states.md
│       └── stoichiometry.md
├── build_test_db.py       # Index all 12 docs into test_sandbox.db
├── retrieve.py            # Retrieve context via E-gating (Born rule)
├── run_all_tests.py       # Run all 4 test problems
├── test_sandbox.db        # Generated database (gitignored)
└── README.md              # This file
```

## Running the Test

### Step 1: Build the Database (Already Done)

```bash
cd THOUGHT/LAB/LIL_Q
python test_sandbox/build_test_db.py
```

This indexes all 12 knowledge base documents into `test_sandbox.db` using the same embedding model as LIL_Q (all-MiniLM-L6-v2).

### Step 2: Run All Tests

```bash
python test_sandbox/run_all_tests.py
```

This runs all 4 problems against both models (big and tiny) with and without context.

### Step 3: Run Individual Problem

```bash
python test_sandbox/run_all_tests.py --problem math
python test_sandbox/run_all_tests.py --problem code
python test_sandbox/run_all_tests.py --problem logic
python test_sandbox/run_all_tests.py --problem chemistry
```

### Step 4: Verbose Mode

```bash
python test_sandbox/run_all_tests.py --verbose
```

Shows full model responses instead of just pass/fail.

## Test Methodology

### Models

- **Big Model**: `qwen2.5-coder:7b` (4.7 GB) - Should solve correctly without context
- **Tiny Model**: `qwen2.5-coder:0.5b` (397 MB) - Should fail without context, succeed with context

### Conditions

**Condition A: No Context** (baseline)
- Big model should succeed (proves problem is solvable)
- Tiny model should fail (proves problem is too hard)

**Condition B: With Context** (quantum rescue)
- Big model should still succeed
- Tiny model should now succeed via quantum entanglement

### Context Retrieval

Context is retrieved using E-gating (Born rule):
```python
E = <query|doc> = dot(query_vec, doc_vec)
```

- Threshold: E ≥ 0.3
- Top k=3 documents per domain
- Domain-filtered (math queries only retrieve math docs, etc.)

### Quantum Entanglement

Context vectors are blended into query state via:
```python
blended_query = query + sum(context_i * E(query, context_i) for context_i in context)
```

This is the quantum entanglement happening on the manifold!

## Success Criteria

**Quantum Rescue** is demonstrated when:
1. Big model succeeds without context (problem is solvable)
2. Tiny model fails without context (problem too hard)
3. Tiny model succeeds with context (quantum rescue!)

**Hypothesis Confirmed** if quantum rescue works in ≥3/4 domains.

## Expected Output

```
======================================================================
QUANTUM ENTANGLEMENT TEST: All 4 Domains
======================================================================

PROBLEM: MATH
...
PROBLEM: CODE
...
PROBLEM: LOGIC
...
PROBLEM: CHEMISTRY
...

======================================================================
RESULTS SUMMARY
======================================================================

| Problem | Big (No Ctx) | Tiny (No Ctx) | Big (Ctx) | Tiny (Ctx) | Rescued? |
|---------|--------------|---------------|-----------|------------|----------|
| math     | PASS         | FAIL          | PASS      | PASS       | YES      |
| code     | PASS         | FAIL          | PASS      | PASS       | YES      |
| logic    | PASS         | FAIL          | PASS      | PASS       | YES      |
| chemistry| PASS         | FAIL          | PASS      | PASS       | YES      |

======================================================================
HYPOTHESIS VALIDATION
======================================================================

Problems where quantum rescue worked: 4/4

*** HYPOTHESIS FULLY CONFIRMED ***
Quantum entanglement with context enabled tiny model across ALL domains!
```

## Technical Details

### E-Gating (Born Rule)

- Formula: E = <psi|phi> = dot(v1, v2)
- Validated by Q44: r=0.977 correlation with quantum Born rule
- Used for context relevance filtering

### Quantum Entanglement

- Formula: `mind_new = ifft(fft(mind) * fft(state))`
- FFT circular convolution in frequency domain
- Updates mind state after each interaction

### Geometric Operations

All operations happen on the semantic manifold:
1. Text → unit vector (enter manifold)
2. E measurement (Born rule)
3. Context blending (weighted sum)
4. Mind entanglement (FFT convolution)
5. LLM generation (exit manifold)

## Why This Test Works

1. **Objective Validation**: Can verify answers mathematically/logically
2. **Known Benchmark**: 7B models can solve, 0.5B models typically can't
3. **Oracle Truth**: Claude knows the answers without context
4. **Teachable**: Solution techniques can be documented clearly
5. **Measurable Rescue**: Clear before/after comparison with ground truth

## Files Created

- 12 knowledge base documents (~300 words each)
- 3 Python scripts (build, retrieve, test)
- 1 database (test_sandbox.db, ~5MB)
- Total: ~4000 lines of test infrastructure

## Clean Up

To remove test artifacts:
```bash
rm test_sandbox/test_sandbox.db
```

The database will be rebuilt on next run of `build_test_db.py`.
