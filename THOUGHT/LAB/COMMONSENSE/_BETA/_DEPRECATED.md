# _v1 - Deprecated Beta (COMMONSENSE v0.2.0)

**Status: DEPRECATED**

This is the original COMMONSENSE tiny model implementation. Replaced by the v2 implementation at the parent directory level.

## Why deprecated

- Phase 2 (symbolic expansion) was structurally broken: `translate.py` read from `codebook["symbols"]` which never existed — legacy @-handles were stored under `codebook["legacy"]`
- No mapping from @-symbols to predicate-level facts (just grammar tokens)
- Single-layer rule matcher with no defeasible defaults, belief revision, or induction

## Preserved for reference

- resolver.py v0.2.0 — Phase 0 (schema) and Phase 1 (resolver) pass clean
- CODEBOOK.json — full ASCII macro grammar (10 radicals, 7 operators, 13 rules, 20 invariants)
- translate.py — original @-symbol expander (broken Phase 2)
- FIXTURES/ — Phase 0/1/2 test fixtures
- TESTBENCH/ — pytest-compatible test suite
- SCHEMAS/ — commonsense_entry and resolution_result JSON schemas

## Migration

New implementation uses proper `codebook["symbols"]` entries and a two-tier expansion: @-handle -> predicates, then predicates -> resolution. See parent `CODEBOOK.json` and `translate.py`.
