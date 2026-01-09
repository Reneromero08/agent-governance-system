#!/usr/bin/env python3
"""
SCL Validator (Phase 5.2.4)

Validates Symbolic Compression Language programs and their expansions.

Validation Layers:
    L1 - Syntax: Well-formed macro notation
    L2 - Symbol: All symbols exist in codebook
    L3 - Semantic: Params match, types valid
    L4 - Expansion: Expanded output passes JobSpec schema

Usage:
    from scl_validator import validate_scl, validate_expansion

    result = validate_scl("C3:build")
    if not result.valid:
        print(result.errors)

    result = validate_expansion({"job_id": "test", ...})
    if not result.valid:
        print(result.errors)
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CODEBOOK_PATH = PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"
JOBSPEC_SCHEMA_PATH = PROJECT_ROOT / "LAW" / "SCHEMAS" / "jobspec.schema.json"

# Add TOOLS to path for codebook_lookup
TOOLS_PATH = PROJECT_ROOT / "CAPABILITY" / "TOOLS"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationResult:
    """Result of SCL validation."""
    valid: bool
    layer: str  # L1, L2, L3, L4
    program: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    parsed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "layer": self.layer,
            "program": self.program,
            "errors": self.errors,
            "warnings": self.warnings,
            "parsed": self.parsed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CODEBOOK CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_codebook_cache = None


def _load_codebook() -> dict:
    """Load and cache the codebook."""
    global _codebook_cache
    if _codebook_cache is not None:
        return _codebook_cache

    if not CODEBOOK_PATH.exists():
        return {}

    with open(CODEBOOK_PATH, "r", encoding="utf-8") as f:
        _codebook_cache = json.load(f)
    return _codebook_cache


def get_known_radicals() -> set:
    """Get set of valid radical symbols."""
    cb = _load_codebook()
    return set(cb.get("radicals", {}).keys())


def get_known_operators() -> set:
    """Get set of valid operator symbols."""
    cb = _load_codebook()
    return set(cb.get("operators", {}).keys())


def get_known_contexts() -> set:
    """Get set of valid context names."""
    cb = _load_codebook()
    return set(cb.get("contexts", {}).keys())


def get_contract_rules() -> set:
    """Get set of valid contract rule numbers."""
    cb = _load_codebook()
    return set(cb.get("contract_rules", {}).keys())


def get_invariants() -> set:
    """Get set of valid invariant numbers."""
    cb = _load_codebook()
    return set(cb.get("invariants", {}).keys())


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SYMBOLS (CJK)
# ═══════════════════════════════════════════════════════════════════════════════

def get_semantic_symbols() -> set:
    """Get set of valid CJK semantic symbols."""
    try:
        from codebook_lookup import SEMANTIC_SYMBOLS
        return set(SEMANTIC_SYMBOLS.keys())
    except ImportError:
        return set()


# ═══════════════════════════════════════════════════════════════════════════════
# L1: SYNTAX VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# Grammar: RADICAL[OPERATOR][NUMBER][:CONTEXT]
# Examples: C3, I5, C*, G, C3:build, C&I, L.C.3

MACRO_PATTERN = re.compile(
    r'^(?P<radical>[A-Z])(?P<operator>[*!?&|.])?(?P<number>\d+)?(?::(?P<context>\w+))?$'
)

# Compound patterns like C&I, L.C.3
COMPOUND_PATTERN = re.compile(
    r'^[A-Z][*!?&|.][A-Z](?:[*!?&|.][A-Z0-9])*(?::\w+)?$'
)

# CJK single character
CJK_PATTERN = re.compile(r'^[\u4e00-\u9fff]$')

# CJK compound pattern (e.g., 法.驗, 證.雜)
CJK_COMPOUND_PATTERN = re.compile(r'^[\u4e00-\u9fff]\.[\u4e00-\u9fff]$')


def validate_syntax(program: str) -> ValidationResult:
    """
    L1 Validation: Check syntax is well-formed.

    Valid forms:
    - Single radical: C, I, V, L, G, S, R, J, A, P
    - Radical + number: C3, I5
    - Radical + operator: C*, V!
    - Radical + operator + number: Not standard
    - Radical + context: C:build, G:audit
    - Full: C3:build
    - Compound: C&I, L.C.3
    - CJK: 法, 真, 契
    """
    program = program.strip()

    if not program:
        return ValidationResult(
            valid=False,
            layer="L1",
            program=program,
            errors=["Empty program"],
        )

    # Check CJK single-token symbols
    if CJK_PATTERN.match(program):
        return ValidationResult(
            valid=True,
            layer="L1",
            program=program,
            parsed={"type": "cjk_symbol", "symbol": program},
        )

    # Check CJK compound patterns (e.g., 法.驗)
    if CJK_COMPOUND_PATTERN.match(program):
        parts = program.split('.')
        return ValidationResult(
            valid=True,
            layer="L1",
            program=program,
            parsed={"type": "cjk_compound", "symbol": program, "parts": parts},
        )

    # Check compound patterns first
    if COMPOUND_PATTERN.match(program):
        return ValidationResult(
            valid=True,
            layer="L1",
            program=program,
            parsed={"type": "compound", "expression": program},
        )

    # Check standard macro pattern
    match = MACRO_PATTERN.match(program)
    if match:
        parsed = {
            "type": "macro",
            "radical": match.group("radical"),
            "operator": match.group("operator"),
            "number": match.group("number"),
            "context": match.group("context"),
        }
        return ValidationResult(
            valid=True,
            layer="L1",
            program=program,
            parsed={k: v for k, v in parsed.items() if v is not None},
        )

    # Invalid syntax
    return ValidationResult(
        valid=False,
        layer="L1",
        program=program,
        errors=[f"Invalid syntax: '{program}' does not match grammar RADICAL[OPERATOR][NUMBER][:CONTEXT]"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# L2: SYMBOL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_symbols(program: str) -> ValidationResult:
    """
    L2 Validation: Check all symbols exist in codebook.
    """
    # First pass L1
    l1_result = validate_syntax(program)
    if not l1_result.valid:
        return l1_result

    errors = []
    warnings = []
    parsed = l1_result.parsed

    # CJK symbols (single or compound)
    if parsed.get("type") in ("cjk_symbol", "cjk_compound"):
        known_cjk = get_semantic_symbols()
        if parsed["symbol"] not in known_cjk:
            errors.append(f"Unknown CJK symbol: '{parsed['symbol']}'. Known: {sorted(known_cjk)[:10]}...")

    # Macro validation
    elif parsed.get("type") == "macro":
        known_radicals = get_known_radicals()
        known_operators = get_known_operators()
        known_contexts = get_known_contexts()

        radical = parsed.get("radical")
        if radical and radical not in known_radicals:
            errors.append(f"Unknown radical: '{radical}'. Known: {sorted(known_radicals)}")

        operator = parsed.get("operator")
        if operator and operator not in known_operators:
            errors.append(f"Unknown operator: '{operator}'. Known: {sorted(known_operators)}")

        context = parsed.get("context")
        if context and context not in known_contexts:
            warnings.append(f"Unknown context: '{context}'. Known: {sorted(known_contexts)}")

        # Check numbered rules exist
        number = parsed.get("number")
        if number and radical:
            if radical == "C":
                known_rules = get_contract_rules()
                rule_key = f"C{number}"
                if rule_key not in known_rules:
                    errors.append(f"Unknown contract rule: '{rule_key}'. Known: {sorted(known_rules)}")
            elif radical == "I":
                known_invs = get_invariants()
                inv_key = f"I{number}"
                if inv_key not in known_invs:
                    errors.append(f"Unknown invariant: '{inv_key}'. Known: {sorted(known_invs)}")

    # Compound expressions - validate each radical
    elif parsed.get("type") == "compound":
        expression = parsed.get("expression", "")
        known_radicals = get_known_radicals()
        known_operators = get_known_operators()

        # Extract all single-letter capitals
        radicals_in_expr = re.findall(r'[A-Z]', expression)
        for r in radicals_in_expr:
            if r not in known_radicals:
                errors.append(f"Unknown radical in compound: '{r}'. Known: {sorted(known_radicals)}")

        # Extract operators
        operators_in_expr = re.findall(r'[*!?&|.]', expression)
        for op in operators_in_expr:
            if op not in known_operators:
                errors.append(f"Unknown operator in compound: '{op}'. Known: {sorted(known_operators)}")

    if errors:
        return ValidationResult(
            valid=False,
            layer="L2",
            program=program,
            errors=errors,
            warnings=warnings,
            parsed=parsed,
        )

    return ValidationResult(
        valid=True,
        layer="L2",
        program=program,
        warnings=warnings,
        parsed=parsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# L3: SEMANTIC VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_semantic(program: str) -> ValidationResult:
    """
    L3 Validation: Check semantic constraints.

    - Operator semantics respected
    - Context applies to radical
    - Numbers in valid range
    """
    # First pass L2
    l2_result = validate_symbols(program)
    if not l2_result.valid:
        return l2_result

    errors = []
    warnings = []
    parsed = l2_result.parsed

    # Semantic checks for macros
    if parsed.get("type") == "macro":
        radical = parsed.get("radical")
        operator = parsed.get("operator")
        number = parsed.get("number")

        # Check * (ALL) operator doesn't have number
        if operator == "*" and number:
            warnings.append(f"Operator '*' (ALL) with number '{number}' is unusual - ignoring number")

        # Check ! (NOT/DENY) semantics
        if operator == "!":
            warnings.append(f"Using denial operator '!' on {radical} - verify intent")

        # Check numbered radicals are C or I
        if number and radical not in ("C", "I"):
            warnings.append(f"Numbered radical {radical}{number} - only C and I have numbered entries")

    if errors:
        return ValidationResult(
            valid=False,
            layer="L3",
            program=program,
            errors=errors,
            warnings=l2_result.warnings + warnings,
            parsed=parsed,
        )

    return ValidationResult(
        valid=True,
        layer="L3",
        program=program,
        warnings=l2_result.warnings + warnings,
        parsed=parsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# L4: EXPANSION VALIDATION (JobSpec Schema)
# ═══════════════════════════════════════════════════════════════════════════════

_jobspec_schema_cache = None


def _load_jobspec_schema() -> dict:
    """Load and cache the JobSpec schema."""
    global _jobspec_schema_cache
    if _jobspec_schema_cache is not None:
        return _jobspec_schema_cache

    if not JOBSPEC_SCHEMA_PATH.exists():
        return {}

    with open(JOBSPEC_SCHEMA_PATH, "r", encoding="utf-8") as f:
        _jobspec_schema_cache = json.load(f)
    return _jobspec_schema_cache


# Allowed output roots (from C8/I6)
ALLOWED_OUTPUT_ROOTS = {
    "_runs/",
    "_generated/",
    "_packs/",
    "_tmp/",
    "INBOX/",
    "NAVIGATION/RECEIPTS/",
}

# Forbidden operations
FORBIDDEN_OPERATIONS = {
    "delete_canon",
    "modify_schema",
    "bypass_verification",
}


def validate_expansion(expansion: dict) -> ValidationResult:
    """
    L4 Validation: Validate expanded JobSpec output.

    Checks:
    - Required fields present
    - Types match schema
    - Output paths in allowed roots
    - No forbidden operations
    """
    errors = []
    warnings = []

    schema = _load_jobspec_schema()
    if not schema:
        warnings.append("JobSpec schema not found - skipping schema validation")

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in expansion:
            errors.append(f"Missing required field: '{field}'")

    # Check job_id format
    job_id = expansion.get("job_id", "")
    if job_id:
        job_id_pattern = schema.get("properties", {}).get("job_id", {}).get("pattern", "")
        if job_id_pattern and not re.match(job_id_pattern, job_id):
            errors.append(f"Invalid job_id format: '{job_id}' (must match {job_id_pattern})")

    # Check task_type enum
    task_type = expansion.get("task_type", "")
    if task_type:
        allowed_types = schema.get("properties", {}).get("task_type", {}).get("enum", [])
        if allowed_types and task_type not in allowed_types:
            errors.append(f"Invalid task_type: '{task_type}'. Allowed: {allowed_types}")

    # Check determinism enum
    determinism = expansion.get("determinism", "")
    if determinism:
        allowed_det = schema.get("properties", {}).get("determinism", {}).get("enum", [])
        if allowed_det and determinism not in allowed_det:
            errors.append(f"Invalid determinism: '{determinism}'. Allowed: {allowed_det}")

    # Check output paths are in allowed roots
    outputs = expansion.get("outputs", {})
    durable_paths = outputs.get("durable_paths", [])
    for path in durable_paths:
        path_ok = any(path.startswith(root) for root in ALLOWED_OUTPUT_ROOTS)
        if not path_ok:
            errors.append(f"Output path '{path}' not in allowed roots: {ALLOWED_OUTPUT_ROOTS}")

    # Check for forbidden operations in metadata
    metadata = expansion.get("metadata", {})
    operations = metadata.get("operations", [])
    for op in operations:
        if op in FORBIDDEN_OPERATIONS:
            errors.append(f"Forbidden operation: '{op}'")

    if errors:
        return ValidationResult(
            valid=False,
            layer="L4",
            program=json.dumps(expansion)[:100] + "...",
            errors=errors,
            warnings=warnings,
            parsed={"expansion": expansion},
        )

    return ValidationResult(
        valid=True,
        layer="L4",
        program=json.dumps(expansion)[:100] + "...",
        warnings=warnings,
        parsed={"expansion": expansion},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN API
# ═══════════════════════════════════════════════════════════════════════════════


def validate_scl(program: str, level: str = "L3") -> ValidationResult:
    """
    Validate an SCL program.

    Args:
        program: SCL program string (e.g., "C3", "法", "C&I:build")
        level: Maximum validation level ("L1", "L2", "L3")

    Returns:
        ValidationResult with valid, errors, warnings, parsed
    """
    validators = {
        "L1": validate_syntax,
        "L2": validate_symbols,
        "L3": validate_semantic,
    }

    if level not in validators:
        return ValidationResult(
            valid=False,
            layer="L0",
            program=program,
            errors=[f"Unknown validation level: '{level}'. Use L1, L2, or L3"],
        )

    return validators[level](program)


def validate_program_list(programs: list, level: str = "L3") -> dict:
    """
    Validate multiple SCL programs.

    Returns:
        {
            "all_valid": bool,
            "results": [ValidationResult.to_dict(), ...],
            "error_count": int,
            "warning_count": int,
        }
    """
    results = []
    error_count = 0
    warning_count = 0

    for prog in programs:
        result = validate_scl(prog, level)
        results.append(result.to_dict())
        if not result.valid:
            error_count += 1
        warning_count += len(result.warnings)

    return {
        "all_valid": error_count == 0,
        "results": results,
        "error_count": error_count,
        "warning_count": warning_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SCL Validator (Phase 5.2.4)")
    parser.add_argument("program", nargs="?", help="SCL program to validate")
    parser.add_argument("--level", choices=["L1", "L2", "L3"], default="L3",
                        help="Validation level (default: L3)")
    parser.add_argument("--jobspec", type=str, help="Path to JobSpec JSON to validate")
    parser.add_argument("--batch", type=str, help="File with programs to validate (one per line)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Validate JobSpec
    if args.jobspec:
        with open(args.jobspec, "r", encoding="utf-8") as f:
            expansion = json.load(f)
        result = validate_expansion(expansion)

    # Batch validate
    elif args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            programs = [line.strip() for line in f if line.strip()]
        result = validate_program_list(programs, args.level)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Validated {len(programs)} programs")
            print(f"Valid: {len(programs) - result['error_count']}/{len(programs)}")
            print(f"Errors: {result['error_count']}, Warnings: {result['warning_count']}")
            for r in result["results"]:
                if not r["valid"]:
                    print(f"  FAIL: {r['program']}: {r['errors']}")
        return 0 if result["all_valid"] else 1

    # Single program
    elif args.program:
        result = validate_scl(args.program, args.level)

    else:
        parser.print_help()
        return 1

    # Output
    if args.json:
        print(json.dumps(result.to_dict() if hasattr(result, 'to_dict') else result,
                         indent=2, ensure_ascii=False))
    else:
        if hasattr(result, 'valid'):
            status = "PASS" if result.valid else "FAIL"
            print(f"{status} [{result.layer}]: {result.program}")
            if result.errors:
                for e in result.errors:
                    print(f"  ERROR: {e}")
            if result.warnings:
                for w in result.warnings:
                    print(f"  WARN: {w}")
            if result.parsed:
                print(f"  Parsed: {result.parsed}")

    return 0 if (hasattr(result, 'valid') and result.valid) else 1


if __name__ == "__main__":
    sys.exit(main())
