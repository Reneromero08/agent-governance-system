---
id: "ADR-035"
title: "Prompt Pack Linter for Canon Enforcement"
status: "Accepted"
date: "2026-01-04"
confidence: "High"
impact: "Medium"
tags: ["governance", "tooling", "prompts", "validation"]
---

# ADR-035: Prompt Pack Linter for Canon Enforcement

## Context

The AGS prompt pack under `NAVIGATION/PROMPTS/` contains generated prompts that must strictly adhere to the canonical policy defined in `1_PROMPT_POLICY_CANON.md`. Manual validation is error-prone and does not scale. Version skew between prompts and canon files can lead to:

1. **Execution failures**: Prompts generated from outdated canon may violate current policy
2. **Governance drift**: Forbidden terms or patterns may slip through without automated detection
3. **Maintenance burden**: Manual verification of YAML frontmatter, hash consistency, and structural requirements is tedious
4. **CI/CD gaps**: No automated gate to prevent non-compliant prompts from being committed

The canon defines specific requirements (sections 4, 8, 11 of `1_PROMPT_POLICY_CANON.md`):
- YAML frontmatter with required fields
- Canon hash consistency for version tracking
- Forbidden inference terms (hex-escaped detection)
- FILL_ME__ token containment
- Manifest and INDEX integrity

Without mechanical enforcement, these requirements rely on human discipline and are vulnerable to drift.

## Decision

Implement a **deterministic, read-only prompt pack linter** at `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh` that mechanically enforces all requirements from `1_PROMPT_POLICY_CANON.md`.

**Key characteristics**:
- **Read-only**: Never modifies any files
- **Deterministic**: Same input always produces same output
- **Strict exit codes**: 
  - 0 = PASS (all checks passed)
  - 1 = POLICY VIOLATION (blocking, must fix)
  - 2 = WARNING (non-blocking, should address)
- **Minimal dependencies**: Bash + Python 3 only (no jq, ripgrep, node)
- **Fast**: <5 seconds typical execution time
- **CI-ready**: Designed for automated pipeline integration

**Checks enforced** (A-G):
- **A) Manifest validity**: JSON structure, tasks array, required fields, path existence
- **B) INDEX link validity**: All markdown links resolve to existing files
- **C) YAML frontmatter**: Required fields, format validation (phase=integer, task_id=N.M format, slug=kebab-case)
- **D) Canon hash consistency**: Prompts must reference current canon file hashes (detects version skew)
- **E) Forbidden terms**: Hex-escaped regex detection for "assume" and "assumption" variants
- **F) Empty bullet lines**: WARNING for lines with only `-` (non-blocking)
- **G) FILL token containment**: `FILL_ME__` tokens only allowed in REQUIRED FACTS section

## Alternatives Considered

### 1. Python-only linter
**Rejected**: Bash wrapper provides better shell integration and exit code handling. Python is used only for complex validation logic.

### 2. JSON Schema validation only
**Rejected**: Cannot enforce all requirements (forbidden terms, hash consistency, link validity). Schema validation is insufficient for full canon compliance.

### 3. Pre-commit hook integration
**Deferred**: Linter is designed to be callable from hooks but not automatically integrated. Allows flexibility in enforcement timing.

### 4. Automatic fixing mode
**Rejected**: Read-only operation is a core principle. Automatic fixes could mask underlying issues and reduce visibility into violations.

## Rationale

**Why this approach**:
1. **Mechanical enforcement**: Removes human error from canon compliance checking
2. **Fast feedback**: <5 seconds execution enables tight development loops
3. **CI/CD integration**: Strict exit codes enable automated gating
4. **Minimal dependencies**: Bash + Python 3 are universally available, no exotic tooling required
5. **Deterministic**: Enables reproducible builds and reliable CI results
6. **Read-only**: Safe to run anywhere, cannot corrupt repository state
7. **Graduated severity**: Exit code 2 (WARNING) allows non-critical issues to be addressed without blocking

**Trade-offs**:
- **No auto-fix**: Violations must be manually corrected (intentional - forces understanding)
- **Bash dependency**: Requires bash shell (acceptable given WSL/Linux/macOS coverage)
- **Python 3 required**: Not pure bash (acceptable - Python 3 is standard in AGS environment)

## Consequences

### Positive
- **Automated canon enforcement**: Prompt pack violations detected immediately
- **Version skew detection**: Canon hash mismatches caught before execution
- **CI/CD gating**: Can block merges on policy violations
- **Documentation**: Comprehensive README, implementation summary, quick reference
- **Testing**: Validation and unit test scripts ensure linter correctness
- **Organized**: Dedicated `linters/` folder for future linter additions

### Negative
- **Initial violations**: Existing prompt pack has hash mismatches (expected - prompts generated from older canon)
- **Maintenance**: Linter must be updated if canon requirements change
- **Learning curve**: Teams must understand exit codes and violation types

### Follow-up Work Required
1. **Resolve existing violations**: Update prompt pack hashes to match current canon OR regenerate prompts
2. **CI integration**: Add linter to GitHub Actions workflow
3. **Pre-commit hook**: Consider adding linter to pre-commit checks for prompt changes
4. **Documentation updates**: Update prompt generation workflow to reference linter

## Enforcement

### Canon Updates
- **`LAW/CANON/CONTRACT.md`**: No changes required (linter is a tool, not a policy)
- **`NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md`**: Already defines requirements (section 7) - linter implements them

### Tooling
- **Location**: `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`
- **Documentation**: 
  - `LINT_PROMPT_PACK_README.md` - Full documentation
  - `LINT_IMPLEMENTATION_SUMMARY.md` - Implementation details
  - `LINT_QUICK_REFERENCE.md` - Quick lookup
- **Testing**:
  - `validate_linter.sh` - Validation demo
  - `test_linter.sh` - Unit tests

### CI/CD Integration (Recommended)
```yaml
- name: Lint Prompt Pack
  run: |
    bash CAPABILITY/TOOLS/linters/lint_prompt_pack.sh NAVIGATION/PROMPTS
    exit_code=$?
    if [ $exit_code -eq 1 ]; then
      echo "❌ Policy violations - blocking"
      exit 1
    elif [ $exit_code -eq 2 ]; then
      echo "⚠️  Warnings detected - review recommended"
      exit 0  # Non-blocking
    fi
```

## Review Triggers

This decision should be revisited if:
1. **Canon requirements change**: New validation rules added to `1_PROMPT_POLICY_CANON.md`
2. **Performance issues**: Linter execution time exceeds 10 seconds
3. **Dependency conflicts**: Bash or Python 3 become unavailable in target environments
4. **False positives**: Linter incorrectly flags valid prompts (indicates bug or spec drift)
5. **Prompt format evolution**: YAML schema or manifest structure changes significantly

## Implementation Status

- ✅ **Linter script**: `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh` (implemented)
- ✅ **Documentation**: Comprehensive README, summary, quick reference (complete)
- ✅ **Testing**: Validation and unit test scripts (complete)
- ✅ **Verification**: Linter operational, detecting real violations (hash mismatches)
- ⏳ **CI integration**: Pending (recommended next step)
- ⏳ **Violation resolution**: Existing prompt pack has hash mismatches (requires regeneration or manual update)
