# TODO: Phase 5 — Translation protocol (minimal executable bundles)

**Roadmap Phase:** Phase 5 (Not Started)

## Tasks

- [ ] Define `Bundle` schema:
  - [ ] intent
  - [ ] refs (symbols)
  - [ ] expand_plan (symbol+slice list)
  - [ ] ops
  - [ ] budgets

- [ ] Implement bundler:
  - [ ] uses discovery to pick candidates
  - [ ] adds only the minimal refs needed
  - [ ] requests explicit expands (sliced) when required

- [ ] Add bundle verifier:
  - [ ] checks budgets
  - [ ] checks all symbols resolvable
  - [ ] checks slice validity

- [ ] Add memoization across steps within a run:
  - [ ] reuse expansions, avoid re-expanding

## Exit Criteria

- [ ] Same task, same corpus → bundles differ only when corpus changes.
- [ ] Measured prompt payload stays small and bounded per step.

---

## References

- Contract: `docs/catalytic-chat/CONTRACT.md`
- Roadmap: `CAT_CHAT_ROADMAP_V1.md` (Phase 5)
