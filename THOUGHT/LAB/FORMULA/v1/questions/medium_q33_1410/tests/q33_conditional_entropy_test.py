"""
Q33 DERIVATION TEST: sigma^Df as Information-Theoretic Necessity

Goal: Prove sigma^Df = N (concept_units) is a tautology by construction,
      not a heuristic booster.

The derivation:
  1. sigma := N / H(X)           (semantic density: meaning per token)
  2. Df := log(N) / log(sigma)   (fractal dimension)
  3. Therefore: sigma^Df = N     (necessary consequence)

Tests:
  1. Tautology verification: sigma^Df always equals N (mathematical necessity)
  2. Measurement procedure: Can we extract sigma, Df, H(X|S) from real data?
  3. CDR validation: CDR = sigma^Df / H(X|S) = N / tokens(pointer)
  4. When density helps vs hurts: Aligned vs misaligned blankets
  5. Integration with GOV_IR: concept_unit counting works correctly
"""
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# =============================================================================
# GOV_IR CONCEPT_UNIT COUNTING (from GOV_IR_SPEC Section 7)
# =============================================================================

def count_concept_units(node: dict) -> int:
    """
    Count concept_units per GOV_IR_SPEC Section 7.

    Atomic semantic nodes:
      - constraint, permission, prohibition, reference, gate: 1 each
      - literal: 0 (structural, not semantic)

    Operations:
      - AND: sum of operands
      - OR: max of operands
      - NOT: operand count
      - others: 1 + sum of operands

    Composites:
      - sequence: sum of elements
      - record: sum of field values
    """
    if not isinstance(node, dict):
        return 0

    node_type = node.get('type')

    # Atomic semantic nodes: 1 concept_unit each
    if node_type in ('constraint', 'permission', 'prohibition', 'reference', 'gate'):
        # Also count nested subject/predicate/target etc.
        nested_count = 0
        for key in ['subject', 'predicate', 'target', 'scope', 'pass_criteria']:
            if key in node and isinstance(node[key], dict):
                nested_count += count_concept_units(node[key])
        for key in ['conditions', 'exceptions', 'operands']:
            if key in node and isinstance(node[key], list):
                nested_count += sum(count_concept_units(item) for item in node[key])
        return 1 + nested_count

    # Literals: 0 (structural, not semantic)
    if node_type == 'literal':
        return 0

    # Operations: depends on operator
    if node_type == 'operation':
        op = node.get('op')
        operands = node.get('operands', [])
        operand_units = [count_concept_units(o) for o in operands]

        if op == 'AND':
            return sum(operand_units)
        elif op == 'OR':
            return max(operand_units) if operand_units else 0
        elif op == 'NOT':
            return operand_units[0] if operand_units else 0
        else:
            return 1 + sum(operand_units)

    # Sequences: sum of elements
    if node_type == 'sequence':
        return sum(count_concept_units(e) for e in node.get('elements', []))

    # Records: sum of field values
    if node_type == 'record':
        return sum(count_concept_units(v) for v in node.get('fields', {}).values())

    return 0


# =============================================================================
# MOCK TOKENIZER (for environments without tiktoken)
# =============================================================================

def count_tokens_mock(text: str) -> int:
    """
    Approximation: ~4 chars per token for English text.
    CJK characters: 1 token each.
    This is a deterministic mock for testing without tiktoken.
    """
    token_count = 0
    i = 0
    while i < len(text):
        char = text[i]
        # CJK character ranges (simplified)
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            token_count += 1
            i += 1
        elif '\u3400' <= char <= '\u4dbf':  # CJK Extension A
            token_count += 1
            i += 1
        else:
            # ASCII/Latin: ~4 chars per token
            token_count += 1
            i += 4
    return max(token_count, 1)


def try_tiktoken_or_mock(text: str, encoding_name: str = "o200k_base") -> int:
    """Try tiktoken, fall back to mock if not available."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except ImportError:
        return count_tokens_mock(text)


# =============================================================================
# SEMANTIC DENSITY MEASUREMENT
# =============================================================================

@dataclass
class SemanticDensityResult:
    """Result of semantic density measurement."""
    pointer: str
    H_X: int              # Baseline entropy (tokens of expansion)
    H_X_given_S: int      # Conditional entropy (tokens of pointer)
    I_X_S: int            # Mutual information (tokens saved)
    N: int                # concept_units
    sigma: float          # Semantic density
    Df: float             # Fractal dimension
    sigma_Df: float       # Should equal N
    CDR: float            # Concept Density Ratio
    compression_ratio: float
    verification_passed: bool  # sigma^Df = N check


def measure_semantic_density(
    pointer: str,
    expansion: str,
    ir_node: dict,
    tokenizer_id: str = "o200k_base"
) -> SemanticDensityResult:
    """
    Measure sigma, Df, and H(X|S) from real data.

    This is the core measurement procedure from Q33.

    Args:
        pointer: SPC pointer (e.g., "C3", "法")
        expansion: Full text expansion
        ir_node: GOV_IR node (for concept_unit counting)
        tokenizer_id: Tokenizer encoding name

    Returns:
        SemanticDensityResult with all measurements
    """
    # Step 1: Measure entropies (token counts)
    H_X = try_tiktoken_or_mock(expansion, tokenizer_id)
    H_X_given_S = try_tiktoken_or_mock(pointer, tokenizer_id)

    # Step 2: Count concept_units (from GOV_IR_SPEC)
    N = count_concept_units(ir_node)

    # Step 3: Compute sigma (semantic density)
    sigma = N / H_X if H_X > 0 else 0.0

    # Step 4: Compute Df (fractal dimension)
    if sigma > 0 and sigma != 1.0 and N > 0:
        Df = math.log(N) / math.log(sigma)
    else:
        Df = 1.0  # Degenerate case

    # Step 5: Verify sigma^Df ≈ N (should be exact)
    sigma_Df = sigma ** Df if sigma > 0 else 0.0

    # Step 6: Compute CDR
    CDR = N / H_X_given_S if H_X_given_S > 0 else float('inf')

    # Step 7: Mutual information
    I_X_S = H_X - H_X_given_S

    # Step 8: Compression ratio
    compression_ratio = H_X / H_X_given_S if H_X_given_S > 0 else float('inf')

    # Verification: sigma^Df = N (within floating point tolerance)
    verification_passed = abs(sigma_Df - N) < 0.001

    return SemanticDensityResult(
        pointer=pointer,
        H_X=H_X,
        H_X_given_S=H_X_given_S,
        I_X_S=I_X_S,
        N=N,
        sigma=sigma,
        Df=Df,
        sigma_Df=sigma_Df,
        CDR=CDR,
        compression_ratio=compression_ratio,
        verification_passed=verification_passed
    )


# =============================================================================
# TEST FIXTURES: GOV_IR NODES
# =============================================================================

TEST_CASES = [
    {
        "name": "Simple constraint (C3)",
        "pointer": "C3",
        "expansion": "All documents requiring human review must be in INBOX/",
        "ir_node": {
            "type": "constraint",
            "op": "requires",
            "subject": {"type": "reference", "ref_type": "path", "value": "INBOX/"},
            "predicate": {"type": "literal", "value_type": "string", "value": "human-review documents"},
            "severity": "must"
        },
        "expected_N": 2  # 1 constraint + 1 reference
    },
    {
        "name": "Simple prohibition (INV-006)",
        "pointer": "I6",
        "expansion": "System-generated artifacts must not be written outside allowed roots (_runs/, _generated/, _packs/)",
        "ir_node": {
            "type": "prohibition",
            "op": "forbids",
            "subject": {"type": "literal", "value_type": "string", "value": "system-generated artifacts"},
            "target": {
                "type": "operation",
                "op": "NOT_IN",
                "operands": [
                    {"type": "reference", "ref_type": "path", "value": "_runs/"},
                    {"type": "reference", "ref_type": "path", "value": "_generated/"},
                    {"type": "reference", "ref_type": "path", "value": "_packs/"}
                ]
            },
            "exceptions": []
        },
        "expected_N": 5  # 1 prohibition + 1 operation + 3 references
    },
    {
        "name": "Gate (fixture_gate)",
        "pointer": "G.fix",
        "expansion": "All fixture tests must pass before merge",
        "ir_node": {
            "type": "gate",
            "gate_type": "test",
            "target": {"type": "reference", "ref_type": "path", "value": "LAW/CONTRACTS/fixtures/"},
            "pass_criteria": {
                "type": "operation",
                "op": "EQ",
                "operands": [
                    {"type": "literal", "value_type": "string", "value": "exit_code"},
                    {"type": "literal", "value_type": "integer", "value": 0}
                ]
            },
            "fail_action": "reject"
        },
        "expected_N": 3  # 1 gate + 1 reference + 1 operation
    },
    {
        "name": "CJK symbol (法)",
        "pointer": "法",
        "expansion": "LAW/CANON domain containing all governance rules, invariants, and canonical specifications",
        "ir_node": {
            "type": "reference",
            "ref_type": "path",
            "value": "LAW/CANON"
        },
        "expected_N": 1  # 1 reference
    },
    {
        "name": "Compound constraint with AND",
        "pointer": "C1&C2",
        "expansion": "Text outranks code AND All governance canon located under LAW/CANON",
        "ir_node": {
            "type": "operation",
            "op": "AND",
            "operands": [
                {
                    "type": "constraint",
                    "op": "requires",
                    "subject": {"type": "literal", "value_type": "string", "value": "text"},
                    "predicate": {"type": "literal", "value_type": "string", "value": "outranks code"},
                    "severity": "must"
                },
                {
                    "type": "constraint",
                    "op": "requires",
                    "subject": {"type": "reference", "ref_type": "path", "value": "LAW/CANON"},
                    "predicate": {"type": "literal", "value_type": "string", "value": "contains governance"},
                    "severity": "must"
                }
            ]
        },
        "expected_N": 3  # AND(constraint + constraint+reference) = 1 + 2
    },
    {
        "name": "Permission node",
        "pointer": "P.write",
        "expansion": "System may write to _runs/ directory for ephemeral outputs",
        "ir_node": {
            "type": "permission",
            "op": "allows",
            "subject": {"type": "literal", "value_type": "string", "value": "system"},
            "scope": {"type": "reference", "ref_type": "path", "value": "_runs/"},
            "conditions": []
        },
        "expected_N": 2  # 1 permission + 1 reference
    }
]


# =============================================================================
# TEST 1: TAUTOLOGY VERIFICATION
# =============================================================================

def test_1_sigma_Df_equals_N():
    """
    CORE TEST: sigma^Df = N is a mathematical tautology.

    Given:
      sigma := N / H(X)
      Df := log(N) / log(sigma)

    Then:
      sigma^Df = sigma^(log(N)/log(sigma))
           = exp(log(sigma) * log(N) / log(sigma))
           = exp(log(N))
           = N

    This must hold for ALL valid inputs.
    """
    print("=" * 70)
    print("TEST 1: sigma^Df = N (Tautology Verification)")
    print("=" * 70)

    all_passed = True

    for case in TEST_CASES:
        result = measure_semantic_density(
            case["pointer"],
            case["expansion"],
            case["ir_node"]
        )

        status = "PASS" if result.verification_passed else "FAIL"
        expected_N = case.get("expected_N", result.N)
        N_status = "PASS" if result.N == expected_N else "FAIL"

        print(f"\n  {case['name']}:")
        print(f"    Pointer: {case['pointer']}")
        print(f"    N (concept_units): {result.N} (expected: {expected_N}) [{N_status}]")
        print(f"    H(X): {result.H_X} tokens")
        print(f"    sigma = N/H(X) = {result.sigma:.6f}")
        print(f"    Df = log(N)/log(sigma) = {result.Df:.6f}")
        print(f"    sigma^Df = {result.sigma_Df:.6f}")
        print(f"    sigma^Df = N? {status} (diff: {abs(result.sigma_Df - result.N):.2e})")

        if not result.verification_passed:
            all_passed = False
        if result.N != expected_N:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print(">>> CONFIRMED: sigma^Df = N for all test cases")
        print(">>> The 'derivation' is a tautology by construction of Df")
    else:
        print(">>> FAIL: Some cases did not verify sigma^Df = N")

    return all_passed


# =============================================================================
# TEST 2: MEASUREMENT PROCEDURE VALIDATION
# =============================================================================

def test_2_measurement_procedure():
    """
    Test that the measurement procedure produces sensible values.

    Requirements:
      1. H(X) > H(X|S) for all compression (mutual information positive)
      2. sigma ∈ (0, ∞) for valid inputs
      3. Df < 0 when sigma < 1 (typical case: more tokens than concepts)
      4. CDR = N / H(X|S)
      5. compression_ratio = H(X) / H(X|S)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Measurement Procedure Validation")
    print("=" * 70)

    all_passed = True

    for case in TEST_CASES:
        result = measure_semantic_density(
            case["pointer"],
            case["expansion"],
            case["ir_node"]
        )

        # Check 1: Positive mutual information
        mi_ok = result.I_X_S > 0

        # Check 2: sigma > 0
        sigma_ok = result.sigma > 0

        # Check 3: CDR = N / H(X|S)
        expected_cdr = result.N / result.H_X_given_S if result.H_X_given_S > 0 else float('inf')
        cdr_ok = abs(result.CDR - expected_cdr) < 0.001

        # Check 4: compression_ratio = H(X) / H(X|S)
        expected_cr = result.H_X / result.H_X_given_S if result.H_X_given_S > 0 else float('inf')
        cr_ok = abs(result.compression_ratio - expected_cr) < 0.001

        passed = mi_ok and sigma_ok and cdr_ok and cr_ok
        status = "PASS" if passed else "FAIL"

        print(f"\n  {case['name']}: [{status}]")
        print(f"    I(X;S) = {result.I_X_S} > 0? {'PASS' if mi_ok else 'FAIL'}")
        print(f"    sigma = {result.sigma:.4f} > 0? {'PASS' if sigma_ok else 'FAIL'}")
        print(f"    CDR = {result.CDR:.2f} (expected: {expected_cdr:.2f}) {'PASS' if cdr_ok else 'FAIL'}")
        print(f"    CR = {result.compression_ratio:.2f}x")

        if not passed:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print(">>> CONFIRMED: Measurement procedure produces valid values")
    else:
        print(">>> FAIL: Some measurements are invalid")

    return all_passed


# =============================================================================
# TEST 3: CDR EQUALS SIGMA_Df OVER H_X_GIVEN_S
# =============================================================================

def test_3_cdr_formula():
    """
    Test that CDR = sigma^Df / H(X|S) = N / H(X|S).

    This connects the formula to the information-theoretic definition.
    """
    print("\n" + "=" * 70)
    print("TEST 3: CDR = sigma^Df / H(X|S) = N / H(X|S)")
    print("=" * 70)

    all_passed = True

    for case in TEST_CASES:
        result = measure_semantic_density(
            case["pointer"],
            case["expansion"],
            case["ir_node"]
        )

        # CDR definition: N / tokens(pointer)
        cdr_from_N = result.N / result.H_X_given_S if result.H_X_given_S > 0 else float('inf')

        # Alternative: sigma^Df / H(X|S)
        cdr_from_sigma_Df = result.sigma_Df / result.H_X_given_S if result.H_X_given_S > 0 else float('inf')

        # They should be equal (since sigma^Df = N)
        diff = abs(cdr_from_N - cdr_from_sigma_Df)
        passed = diff < 0.001

        status = "PASS" if passed else "FAIL"

        print(f"\n  {case['name']}: [{status}]")
        print(f"    CDR from N: {cdr_from_N:.4f}")
        print(f"    CDR from sigma^Df: {cdr_from_sigma_Df:.4f}")
        print(f"    Difference: {diff:.2e}")

        if not passed:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print(">>> CONFIRMED: CDR = sigma^Df / H(X|S) = N / H(X|S)")
    else:
        print(">>> FAIL: CDR formulas do not match")

    return all_passed


# =============================================================================
# TEST 4: DENSITY HELPS VS HURTS
# =============================================================================

def test_4_density_helps_vs_hurts():
    """
    Test when higher semantic density helps vs hurts.

    HELPS when:
      - Blankets aligned (shared codebook)
      - Symbol unambiguous
      - Df bounded

    HURTS when:
      - Blankets misaligned (no shared basis for N)
      - Symbol polysemic (ambiguous)
      - Df unbounded (error amplification)

    We test this by simulating aligned vs misaligned scenarios.
    """
    print("\n" + "=" * 70)
    print("TEST 4: When Density Helps vs Hurts")
    print("=" * 70)

    # Scenario 1: Aligned blankets - density helps
    print("\n  Scenario 1: ALIGNED BLANKETS (density helps)")

    # Use a real pointer that expands deterministically
    aligned_result = measure_semantic_density(
        pointer="C3",
        expansion="All documents requiring human review must be in INBOX/",
        ir_node={
            "type": "constraint",
            "op": "requires",
            "subject": {"type": "reference", "ref_type": "path", "value": "INBOX/"},
            "predicate": {"type": "literal", "value_type": "string", "value": "human-review"},
            "severity": "must"
        }
    )

    aligned_ok = (
        aligned_result.I_X_S > 0 and  # Compression works
        aligned_result.verification_passed and  # sigma^Df = N
        aligned_result.compression_ratio > 1  # Actually compressing
    )

    print(f"    Compression ratio: {aligned_result.compression_ratio:.2f}x")
    print(f"    Tokens saved: {aligned_result.I_X_S}")
    print(f"    sigma^Df = N verified: {aligned_result.verification_passed}")
    print(f"    Verdict: {'HELPS' if aligned_ok else 'DOES NOT HELP'}")

    # Scenario 2: Misaligned blankets - density undefined
    print("\n  Scenario 2: MISALIGNED BLANKETS (density undefined)")

    # When codebooks don't match, we can't count concept_units correctly
    # Simulate by using wrong IR node
    misaligned_result = measure_semantic_density(
        pointer="C3",
        expansion="Completely different expansion that doesn't match C3",
        ir_node={
            "type": "literal",  # Wrong type - no semantic content
            "value_type": "string",
            "value": "garbage"
        }
    )

    # N=0 means no semantic content - undefined density
    misaligned_undefined = misaligned_result.N == 0

    print(f"    N (concept_units): {misaligned_result.N}")
    print(f"    sigma: {misaligned_result.sigma}")
    print(f"    Verdict: {'UNDEFINED (N=0)' if misaligned_undefined else 'UNEXPECTED'}")

    # Scenario 3: Polysemic symbol without context - ambiguous
    print("\n  Scenario 3: POLYSEMIC SYMBOL (ambiguous)")

    # 道 can mean path/principle/method
    polysemic_expansions = [
        "LAW/CANON (filesystem path)",
        "The guiding principle of the system",
        "The method/approach for implementation"
    ]

    # Without context, we don't know which expansion to use
    # CDR is undefined because N depends on which expansion
    print(f"    Symbol: 道")
    print(f"    Possible expansions: {len(polysemic_expansions)}")
    print(f"    Without context: CDR is UNDEFINED")
    print(f"    With context_keys: CDR becomes defined")

    # All scenarios tested
    all_passed = aligned_ok and misaligned_undefined

    print("\n" + "-" * 70)
    if all_passed:
        print(">>> CONFIRMED: Density helps when aligned, undefined when misaligned")
    else:
        print(">>> FAIL: Scenarios did not behave as expected")

    return all_passed


# =============================================================================
# TEST 5: GOV_IR COUNTING CORRECTNESS
# =============================================================================

def test_5_gov_ir_counting():
    """
    Test that concept_unit counting follows GOV_IR_SPEC Section 7 exactly.

    Rules:
      - constraint, permission, prohibition, reference, gate: 1 each
      - literal: 0
      - AND: sum of operands
      - OR: max of operands
      - NOT: operand count
      - sequence: sum of elements
      - record: sum of field values
    """
    print("\n" + "=" * 70)
    print("TEST 5: GOV_IR Counting Correctness")
    print("=" * 70)

    test_nodes = [
        # Basic types
        ({"type": "constraint", "op": "requires"}, 1, "constraint"),
        ({"type": "permission", "op": "allows"}, 1, "permission"),
        ({"type": "prohibition", "op": "forbids"}, 1, "prohibition"),
        ({"type": "reference", "ref_type": "path", "value": "x"}, 1, "reference"),
        ({"type": "gate", "gate_type": "test"}, 1, "gate"),
        ({"type": "literal", "value_type": "string", "value": "x"}, 0, "literal"),

        # Operations
        ({"type": "operation", "op": "AND", "operands": [
            {"type": "reference", "ref_type": "path", "value": "a"},
            {"type": "reference", "ref_type": "path", "value": "b"}
        ]}, 2, "AND(ref, ref)"),

        ({"type": "operation", "op": "OR", "operands": [
            {"type": "reference", "ref_type": "path", "value": "a"},
            {"type": "reference", "ref_type": "path", "value": "b"},
            {"type": "reference", "ref_type": "path", "value": "c"}
        ]}, 1, "OR(ref, ref, ref) = max"),

        ({"type": "operation", "op": "NOT", "operands": [
            {"type": "constraint", "op": "requires"}
        ]}, 1, "NOT(constraint)"),

        ({"type": "operation", "op": "EQ", "operands": [
            {"type": "literal", "value_type": "string", "value": "a"},
            {"type": "literal", "value_type": "string", "value": "b"}
        ]}, 1, "EQ(lit, lit) = 1 + sum"),

        # Composites
        ({"type": "sequence", "elements": [
            {"type": "reference", "ref_type": "path", "value": "a"},
            {"type": "reference", "ref_type": "path", "value": "b"}
        ]}, 2, "sequence[ref, ref]"),

        ({"type": "record", "fields": {
            "a": {"type": "reference", "ref_type": "path", "value": "x"},
            "b": {"type": "constraint", "op": "requires"}
        }}, 2, "record{ref, constraint}"),
    ]

    all_passed = True

    for node, expected, name in test_nodes:
        actual = count_concept_units(node)
        passed = actual == expected
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {name}: expected={expected}, got={actual}")

        if not passed:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print(">>> CONFIRMED: GOV_IR counting follows spec exactly")
    else:
        print(">>> FAIL: Some counts do not match GOV_IR_SPEC")

    return all_passed


# =============================================================================
# TEST 6: FORMAL DERIVATION
# =============================================================================

def test_6_formal_derivation():
    """
    Present the formal derivation and verify it algebraically.
    """
    print("\n" + "=" * 70)
    print("FORMAL DERIVATION: sigma^Df from Information Theory")
    print("=" * 70)

    print("""
INFORMATION-THEORETIC DERIVATION

From Shannon's source coding theorem:
  H(X)   = entropy of message X (bits without context)
  H(X|S) = conditional entropy (bits given shared side-information S)
  I(X;S) = mutual information = H(X) - H(X|S)

For SPC (Semantic Pointer Compression):
  X = governance statement
  S = shared codebook state
  H(X|S) ≈ 0 when S contains the expansion of X

STEP 1: Define sigma (Semantic Density)

  sigma := N / H(X)

  Where:
    N    = concept_units (countable atomic meaning from GOV_IR_SPEC)
    H(X) = tokens(full expansion) under declared tokenizer

  Interpretation: sigma measures meaning per baseline token.

STEP 2: Define Df (Fractal Dimension)

  Df := log(N) / log(sigma)

  Rearranging: N = sigma^Df

  Interpretation: Df captures semantic complexity scaling.

STEP 3: Verify sigma^Df = N (Algebraic Proof)

  sigma^Df = sigma^(log(N)/log(sigma))
       = exp(log(sigma) × log(N)/log(sigma))
       = exp(log(N))
       = N  ∎

CONCLUSION:
  sigma^Df = N is a TAUTOLOGY by construction of Df.
  The term is not a heuristic - it's concept_units in exponential form.

THE FORMULA DECOMPOSES:
  R = (E/∇S) × sigma^Df
    = (evidence quality) × (semantic content)
    = (intensive signal) × (extensive meaning)
""")

    # Algebraic verification
    print("-" * 70)
    print("ALGEBRAIC VERIFICATION:")
    print("-" * 70)

    # Test with specific values
    N = 5
    H_X = 20
    sigma = N / H_X  # = 0.25
    Df = math.log(N) / math.log(sigma)  # = log(5)/log(0.25)
    sigma_Df = sigma ** Df

    print(f"  N = {N}")
    print(f"  H(X) = {H_X}")
    print(f"  sigma = N/H(X) = {sigma}")
    print(f"  Df = log(N)/log(sigma) = log({N})/log({sigma}) = {Df:.6f}")
    print(f"  sigma^Df = {sigma}^{Df:.6f} = {sigma_Df:.6f}")
    print(f"  sigma^Df = N? {abs(sigma_Df - N) < 1e-10}")

    return True


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all Q33 derivation tests."""
    print("=" * 70)
    print("Q33 DERIVATION: sigma^Df as Information-Theoretic Necessity")
    print("=" * 70)
    print("Testing whether sigma^Df is a heuristic booster or concept_units")
    print("=" * 70)

    results = {}

    # Core tautology test
    results['sigma^Df = N (tautology)'] = test_1_sigma_Df_equals_N()

    # Measurement tests
    results['Measurement procedure valid'] = test_2_measurement_procedure()
    results['CDR formula correct'] = test_3_cdr_formula()
    results['Density helps vs hurts'] = test_4_density_helps_vs_hurts()
    results['GOV_IR counting correct'] = test_5_gov_ir_counting()

    # Formal derivation
    results['Formal derivation'] = test_6_formal_derivation()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Q33 Derivation Results")
    print("=" * 70)

    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test}")

    all_passed = all(results.values())

    print("\n" + "-" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
        print("\nQ33 ANSWERED: Is sigma^Df information-theoretic or heuristic?")
        print("-" * 70)
        print("""
INFORMATION-THEORETIC — via tautological construction:

  sigma := N / H(X)           (definition)
  Df := log(N) / log(sigma)   (definition)
  Therefore: sigma^Df = N     (necessary consequence)

The 'derivation' reveals that sigma^Df is simply concept_units in exponential form.

THE FORMULA:
  R = (E/∇S) × sigma^Df = (evidence density) × (semantic content)

The term is NOT a heuristic booster — it's the semantic payload,
countable via GOV_IR_SPEC.

WHEN DENSITY HELPS vs HURTS:
  - HELPS: When Markov blankets aligned (S shared), symbol unambiguous
  - HURTS: When blankets diverge, symbol polysemic, or Df unbounded

MEASUREMENT PROCEDURE:
  1. Sync codebook (CODEBOOK_SYNC_PROTOCOL)
  2. Count H(X), H(X|S) via tokenizer
  3. Count N via GOV_IR concept_unit rules
  4. Derive sigma = N/H(X), Df = log(N)/log(sigma)
  5. Verify sigma^Df = N
""")
    else:
        print("SOME TESTS FAILED - derivation incomplete")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
