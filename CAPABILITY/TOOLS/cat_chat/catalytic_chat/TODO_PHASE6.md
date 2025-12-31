# TODO: Phase 6 — Measurement and regression harness

**Roadmap Phase:** Phase 6 (Not Started)

## Tasks

- [ ] Log per-step metrics:
  - [ ] tokens_in/tokens_out (if available)
  - [ ] bytes_expanded
  - [ ] expands_per_step
  - [ ] reuse_rate
  - [ ] search_k and hit-rate (when ground-truth available)

- [ ] Add regression tests:
  - [ ] determinism tests for SECTION_INDEX + SYMBOLS
  - [ ] budget enforcement tests
  - [ ] receipt completeness tests

- [ ] Add benchmark scenarios:
  - [ ] "find and patch 1 function" task
  - [ ] "refactor N files" task
  - [ ] "generate roadmap from corpus" task

## Exit Criteria

- [ ] A dashboard (or printed report) shows token and expansion savings over baseline.
- [ ] Regressions fail tests deterministically.

---

## References

- Contract: `docs/catalytic-chat/CONTRACT.md`
- Roadmap: `CAT_CHAT_ROADMAP_V1.md` (Phase 6)
