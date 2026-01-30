---
name: test-primitives
version: "0.1.0"
description: Test CAT LAB safe primitives including file locking, atomic writes, and task validation
compatibility: all
---

**required_canon_version:** >=3.0.0

# Skill: test-primitives

**Version:** 0.1.0

**Status:** Active

## Purpose

Tests the safe primitives used throughout the AGS system, including:
- Platform-specific file locking (Windows/Unix)
- Atomic JSONL write and rewrite operations
- Streaming JSONL reader
- Task state transition validation
- Task spec validation
- File hashing (SHA-256)
- Validator constants verification

## Trigger

Use when:
- Verifying that safe primitives work correctly on the current platform
- After system updates that might affect file operations
- During CI/CD pipeline validation
- Testing cross-platform compatibility

## Inputs

JSON object with optional fields:
- `run_tests` (boolean, optional): Whether to run actual tests (default: true)
- `dry_run` (boolean, optional): Return mock results for deterministic testing

## Outputs

JSON object:
- `tests_run` (integer): Number of tests executed
- `tests_passed` (integer): Number of tests that passed
- `tests_failed` (integer): Number of tests that failed
- `details` (array): List of test results with pass/fail indicators
- `summary` (string): Human-readable summary
- `validator_semver` (string): Current validator semantic version
- `build_id` (string): Validator build fingerprint
- `durable_roots_count` (integer): Number of durable roots
- `catalytic_roots_count` (integer): Number of catalytic roots

## Constraints

- Creates temporary files during test execution
- All temporary files are cleaned up after tests
- Platform-specific behavior for file locking
- Non-destructive (does not modify repository files)

## Tests Performed

1. **File locking (Windows/Unix)** - Exclusive file lock acquire/release
2. **Atomic JSONL write** - Append lines atomically
3. **Atomic JSONL rewrite** - Transform and rewrite JSONL atomically
4. **Streaming JSONL reader** - Read with pagination and filtering
5. **Task state transitions** - Validate allowed state changes
6. **Task spec validation** - Validate task specification format
7. **File hashing (SHA-256)** - Compute file hashes
8. **Constants and build ID** - Verify validator constants

## Fixtures

- `fixtures/basic/` - Deterministic test with mock results

**required_canon_version:** >=3.0.0
